// Before the app build: build the console CLI and stage it as a Tauri sidecar
// (binaries/rt-captions-<triple>[.exe]). tauri-build insists the sidecar exists while the crate
// compiles, so a placeholder goes in first and the real CLI replaces it.
import { execSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
process.chdir(fileURLToPath(new URL("..", import.meta.url))); // src-tauri, wherever we are run from
const triple = execSync("rustc -vV").toString().match(/host: (\S+)/)[1];
const ext = process.platform === "win32" ? ".exe" : "";
const sidecar = `binaries/rt-captions-${triple}${ext}`;
mkdirSync("binaries", { recursive: true });
if (!existsSync(sidecar)) writeFileSync(sidecar, "");
execSync("cargo build --release --bin rt-captions", { stdio: "inherit" });
copyFileSync(`target/release/rt-captions${ext}`, sidecar);
