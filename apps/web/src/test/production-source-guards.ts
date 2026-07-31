import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";

const SOURCE_EXTENSION = /\.(?:css|tsx?)$/;
const DECLARATION_FILE = /\.d\.ts$/;
const TEST_FILE = /\.(?:test|spec)\.(?:tsx?|css)$/;
const TEST_DIRECTORIES = new Set(["test", "__tests__"]);

function productionFiles(root: string): string[] {
  function walk(directory: string): string[] {
    return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
      if (entry.isDirectory()) {
        return TEST_DIRECTORIES.has(entry.name)
          ? []
          : walk(path.join(directory, entry.name));
      }
      if (
        !SOURCE_EXTENSION.test(entry.name) ||
        DECLARATION_FILE.test(entry.name) ||
        TEST_FILE.test(entry.name)
      ) {
        return [];
      }
      return [path.join(directory, entry.name)];
    });
  }

  return walk(root).sort();
}

function relative(root: string, file: string): string {
  return path.relative(root, file).split(path.sep).join("/");
}

/** Remove comments without changing quoted import paths or source offsets. */
function withoutComments(source: string): string {
  const output = source.split("");
  let quote: "'" | '"' | "`" | null = null;
  let index = 0;
  while (index < source.length) {
    const current = source[index];
    const next = source[index + 1];
    if (quote !== null) {
      if (current === "\\") {
        index += 2;
        continue;
      }
      if (current === quote) quote = null;
      index += 1;
      continue;
    }
    if (current === "'" || current === '"' || current === "`") {
      quote = current;
      index += 1;
      continue;
    }
    if (current === "/" && next === "/") {
      output[index] = " ";
      output[index + 1] = " ";
      index += 2;
      while (index < source.length && source[index] !== "\n") {
        output[index] = " ";
        index += 1;
      }
      continue;
    }
    if (current === "/" && next === "*") {
      output[index] = " ";
      output[index + 1] = " ";
      index += 2;
      while (
        index < source.length &&
        !(source[index] === "*" && source[index + 1] === "/")
      ) {
        if (source[index] !== "\n") output[index] = " ";
        index += 1;
      }
      if (index < source.length) {
        output[index] = " ";
        output[index + 1] = " ";
        index += 2;
      }
      continue;
    }
    index += 1;
  }
  return output.join("");
}

function localSpecifiers(file: string): string[] {
  const source = withoutComments(readFileSync(file, "utf8"));
  const specifiers: string[] = [];
  const patterns = [
    /\b(?:import|export)\s+(?:type\s+)?(?:[^"'();]*?\s+from\s+)?["']([^"']+)["']/g,
    /\bimport\s*\(\s*["']([^"']+)["']\s*\)/g,
    /@import\s+(?:url\(\s*)?["']([^"']+)["']/g,
  ];
  for (const pattern of patterns) {
    for (const match of source.matchAll(pattern)) {
      const specifier = match[1];
      if (specifier?.startsWith(".")) specifiers.push(specifier);
    }
  }
  return specifiers;
}

function resolveLocalImport(
  importer: string,
  specifier: string,
  production: Set<string>,
): string | null {
  const clean = specifier.split(/[?#]/, 1)[0];
  if (!clean) return null;
  const target = path.resolve(path.dirname(importer), clean);
  const extension = path.extname(target);
  const candidates = extension
    ? [target]
    : [
        target,
        `${target}.ts`,
        `${target}.tsx`,
        `${target}.css`,
        path.join(target, "index.ts"),
        path.join(target, "index.tsx"),
        path.join(target, "index.css"),
      ];
  return candidates.find((candidate) => production.has(candidate)) ?? null;
}

export function findUnreachableProductionFiles(
  root: string,
  entry = "main.tsx",
): string[] {
  const absoluteRoot = path.resolve(root);
  const files = productionFiles(absoluteRoot);
  const production = new Set(files);
  const entryFile = path.resolve(absoluteRoot, entry);
  if (!production.has(entryFile)) {
    throw new Error(`Production entry is not a source file: ${entry}`);
  }

  const reachable = new Set<string>();
  const pending = [entryFile];
  while (pending.length > 0) {
    const file = pending.pop();
    if (file === undefined || reachable.has(file)) continue;
    reachable.add(file);
    for (const specifier of localSpecifiers(file)) {
      const resolved = resolveLocalImport(file, specifier, production);
      if (resolved !== null && !reachable.has(resolved)) pending.push(resolved);
    }
  }

  return files
    .filter((file) => !reachable.has(file))
    .map((file) => relative(absoluteRoot, file));
}

function isRawBlack(value: string): boolean {
  const compact = value.toLowerCase().replaceAll("_", " ");
  return (
    /(?:^|[^a-z])black(?:$|[^a-z])/.test(compact) ||
    /#(?:000[0-9a-f]|000000[0-9a-f]{2}|000|000000)(?![0-9a-f])/.test(
      compact,
    ) ||
    /rgba?\(\s*0(?:\s*,\s*|\s+)0(?:\s*,\s*|\s+)0(?:\s*[,/]\s*[^)]*)?\)/.test(
      compact,
    )
  );
}

function tagAround(source: string, at: number): string {
  const start = source.lastIndexOf("<", at);
  const end = source.indexOf(">", at);
  if (start === -1 || end === -1) {
    return source.slice(Math.max(0, at - 240), at + 240);
  }
  return source.slice(start, end + 1);
}

function isOverlayContext(source: string, at: number): boolean {
  const tag = tagAround(source, at);
  const nearby = source.slice(Math.max(0, at - 180), at + 180);
  const context = `${tag}\n${nearby}`;
  if (/\b(?:modal|dialog|overlay|backdrop|scrim)\b/i.test(context)) return true;
  return (
    /\bfixed\b/.test(context) &&
    (/\binset-0\b/.test(context) ||
      /\binset\s*:\s*(?:0|["']0["'])/.test(context))
  );
}

function cssBlock(source: string, at: number): { body: string; selector: string } {
  const open = source.lastIndexOf("{", at);
  const close = source.indexOf("}", at);
  const boundary = Math.max(
    source.lastIndexOf("}", open - 1),
    source.lastIndexOf("{", open - 1),
  );
  return {
    selector: source.slice(boundary + 1, open).replace(/\s+/g, " ").trim(),
    body: source.slice(open + 1, close === -1 ? source.length : close),
  };
}

function isOverlayCssRule(selector: string, body: string): boolean {
  if (/(?:^|[-_.#])(?:modal|dialog|overlay|backdrop|scrim)(?:$|[-_.:\s#])/i.test(selector)) {
    return true;
  }
  return (
    /position\s*:\s*fixed\b/i.test(body) &&
    /inset\s*:\s*0(?:\D|$)/i.test(body)
  );
}

export function findScrimBypasses(root: string): string[] {
  const absoluteRoot = path.resolve(root);
  const bypasses: string[] = [];
  const allowedTokenDefinitions = new Set([
    "--color-scrim",
    "--color-scrim-strong",
  ]);
  const seenAllowedDefinitions = new Set<string>();
  for (const file of productionFiles(absoluteRoot)) {
    const source = withoutComments(readFileSync(file, "utf8"));
    const name = relative(absoluteRoot, file);
    if (/\.tsx?$/.test(file)) {
      const utilityPattern =
        /\bbg-(?:black(?:\/[^\s"'`]+)?|\[[^\]\s]+\](?:\/[^\s"'`]+)?)/g;
      for (const match of source.matchAll(utilityPattern)) {
        const utility = match[0];
        if (
          match.index !== undefined &&
          isRawBlack(utility) &&
          isOverlayContext(source, match.index)
        ) {
          bypasses.push(`${name}:${utility}`);
        }
      }

      const inlinePattern =
        /(?:background(?:Color)?|["']background-color["'])\s*:\s*(["'`])([^"'`]+)\1/g;
      for (const match of source.matchAll(inlinePattern)) {
        if (
          match.index !== undefined &&
          isRawBlack(match[2] ?? "") &&
          isOverlayContext(source, match.index)
        ) {
          bypasses.push(`${name}:${match[0]}`);
        }
      }
      continue;
    }

    const declarationPattern =
      /(?:^|[;{])\s*(background(?:-color)?)\s*:\s*([^;}]+)/gim;
    for (const match of source.matchAll(declarationPattern)) {
      if (match.index === undefined || !isRawBlack(match[2] ?? "")) continue;
      const block = cssBlock(source, match.index);
      if (!isOverlayCssRule(block.selector, block.body)) continue;
      const declaration = `${match[1]}: ${(match[2] ?? "").trim()}`;
      bypasses.push(`${name}:${block.selector}:${declaration}`);
    }

    const tokenPattern =
      /(?:^|[;{])\s*(--[\w-]*(?:scrim|backdrop)[\w-]*)\s*:\s*([^;}]+)/gim;
    for (const match of source.matchAll(tokenPattern)) {
      const property = match[1] ?? "";
      const value = match[2] ?? "";
      if (!isRawBlack(value) || match.index === undefined) continue;
      const isAllowed =
        path.basename(file) === "design-tokens.css" &&
        allowedTokenDefinitions.has(property) &&
        !seenAllowedDefinitions.has(property);
      if (isAllowed) {
        seenAllowedDefinitions.add(property);
        continue;
      }
      const block = cssBlock(source, match.index);
      bypasses.push(
        `${name}:${block.selector}:${property}: ${value.trim()}`,
      );
    }
  }
  return bypasses;
}

export function findAttributionSelectorBypasses(root: string): string[] {
  const absoluteRoot = path.resolve(root);
  const bypasses: string[] = [];
  for (const file of productionFiles(absoluteRoot)) {
    if (!/\.(?:css|tsx)$/.test(file)) continue;
    const source = withoutComments(readFileSync(file, "utf8"));
    for (const match of source.matchAll(/\.react-flow__attribution\b/g)) {
      bypasses.push(
        `${relative(absoluteRoot, file)}:${match[0]}`,
      );
    }
  }
  return bypasses;
}
