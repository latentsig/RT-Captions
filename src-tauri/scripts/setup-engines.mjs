// Builds/fetches the native speech-engine libraries the app bundles, for this OS:
//   engine/macos/   libparakeet.dylib
//   engine/windows/ libparakeet.dll (+ libc++.dll, libomp.dll, libunwind.dll)
//
//   node src-tauri/scripts/setup-engines.mjs [--force]
//
// Needs: git, cmake, ninja (pip install cmake ninja, or brew install cmake ninja).
// macOS: Xcode command line tools (xcode-select --install).
// Windows: the llvm-mingw clang toolchain is downloaded automatically (MSVC builds of parakeet.cpp
// run its ternary kernel ~17x slower, and its master does not compile with MSVC).
//
// Builds are PORTABLE (GGML_NATIVE=OFF): a native build only runs on CPUs with the build machine's
// instruction set (an M3 build can crash on an M1; an Alder Lake build on older Intel).

import { execFileSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, readdirSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// Our fork of mudler/parakeet.cpp, branch rt-captions: upstream master @5c9b0fb plus our patches
// (streaming CTC decoding for hybrid models, RNN-T/CTC blank penalties; see the branch log).
const PARAKEET_REPO = "https://github.com/latentsig/parakeet.cpp";
const PARAKEET_COMMIT = "1e2085b8e6fae1eff81c32cddd1abfb2bedb0753"; // branch rt-captions
const LLVM_MINGW = "llvm-mingw-20260922-ucrt-x86_64";

const root = fileURLToPath(new URL("..", import.meta.url)); // src-tauri
const force = process.argv.includes("--force");
const mac = process.platform === "darwin";
const win = process.platform === "win32";
if (!mac && !win) throw new Error("setup-engines supports macOS and Windows");
if (mac && process.arch !== "arm64") console.warn("warning: Intel Macs are untested; building anyway");

const out = join(root, "engine", mac ? "macos" : "windows");
const work = join(root, "engine", ".build");
mkdirSync(out, { recursive: true });
mkdirSync(work, { recursive: true });

const run = (cmd, args, opts = {}) => execFileSync(cmd, args, { stdio: "inherit", ...opts });
// Windows' own bsdtar reads zip files; a GNU tar from Git Bash on PATH would not (and reads "C:" as a host).
const TAR = win ? join(process.env.SystemRoot || "C:\\Windows", "System32", "tar.exe") : "tar";

async function download(url, dest) {
  console.log(`downloading ${url}`);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  writeFileSync(dest, Buffer.from(await res.arrayBuffer()));
}

// parakeet.cpp (runs Nemotron, Parakeet Redux/Ultra): built from source at a pinned commit.
const parakeetOut = join(out, mac ? "libparakeet.dylib" : "libparakeet.dll");
if (force || !existsSync(parakeetOut)) {
  const src = join(work, "parakeet.cpp");
  if (!existsSync(join(src, ".git"))) run("git", ["clone", PARAKEET_REPO, src]);
  run("git", ["remote", "set-url", "origin", PARAKEET_REPO], { cwd: src }); // older checkouts point at upstream
  run("git", ["fetch", "origin", PARAKEET_COMMIT], { cwd: src });
  run("git", ["checkout", "--force", PARAKEET_COMMIT], { cwd: src });
  run("git", ["submodule", "update", "--init", "--recursive"], { cwd: src });

  const env = { ...process.env };
  let toolBin = null;
  if (win) {
    toolBin = join(work, LLVM_MINGW, "bin");
    if (!existsSync(join(toolBin, "clang.exe"))) {
      const zip = join(work, "llvm-mingw.zip");
      await download(`https://github.com/mstorsjo/llvm-mingw/releases/download/20260922/${LLVM_MINGW}.zip`, zip);
      run(TAR, ["-xf", "llvm-mingw.zip"], { cwd: work });
      rmSync(zip);
    }
    env.PATH = `${toolBin};${env.PATH}`;
    env.CC = join(toolBin, "clang.exe");
    env.CXX = join(toolBin, "clang++.exe");
  }

  const build = join(src, "build-release");
  if (force) rmSync(build, { recursive: true, force: true });
  const flags = [
    "-S", src, "-B", build, "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DPARAKEET_SHARED=ON", "-DPARAKEET_BUILD_CLI=OFF", "-DPARAKEET_BUILD_SERVER=OFF", "-DPARAKEET_BUILD_TESTS=OFF",
    // Keep GGML inside libparakeet so the copied library has no private build-tree dependencies.
    "-DGGML_NATIVE=OFF", "-DBUILD_SHARED_LIBS=OFF",
    ...(mac
      // GGML's separate BLAS backend uses Accelerate APIs added in macOS 13.3. The CPU backend
      // still uses Accelerate, while disabling BLAS preserves the app's declared macOS 12 floor.
      ? ["-DPARAKEET_GGML_METAL=ON", "-DGGML_METAL_EMBED_LIBRARY=ON", "-DGGML_BLAS=OFF", `-DCMAKE_OSX_ARCHITECTURES=${process.arch === "arm64" ? "arm64" : "x86_64"}`, "-DCMAKE_OSX_DEPLOYMENT_TARGET=12.0"]
      : ["-DGGML_AVX=ON", "-DGGML_AVX2=ON", "-DGGML_FMA=ON", "-DGGML_F16C=ON"]), // AVX2 baseline: Intel 2013+ / AMD Zen
  ];
  run("cmake", flags, { env });
  run("cmake", ["--build", build, "--target", "parakeet", "--parallel"], { env });

  const lib = readdirSync(build, { recursive: true }).map(String)
    .find((f) => f.endsWith(mac ? "libparakeet.dylib" : "libparakeet.dll"));
  if (!lib) throw new Error("libparakeet was not produced; check the build log above");
  cpSync(join(build, lib), parakeetOut, { dereference: true });
  if (mac) run("install_name_tool", ["-id", "@rpath/libparakeet.dylib", parakeetOut]);
  if (win) for (const dll of ["libc++.dll", "libomp.dll", "libunwind.dll"]) cpSync(join(toolBin, dll), join(out, dll));
  console.log(`-> ${parakeetOut}`);
}

console.log(`\nEngine libraries ready in ${resolve(out)}:\n  ${readdirSync(out).join("\n  ")}`);
