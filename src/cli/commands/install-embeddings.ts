/**
 * AgentDB Install Embeddings Command
 * Install optional embedding dependencies (@xenova/transformers + onnxruntime)
 */

import { spawnSync } from 'child_process';

// Color codes for beautiful output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  red: '\x1b[31m',
  cyan: '\x1b[36m',
  magenta: '\x1b[35m'
};

interface InstallEmbeddingsOptions {
  global?: boolean;
}

export async function installEmbeddingsCommand(options: InstallEmbeddingsOptions = {}): Promise<void> {
  console.log(`\n${colors.bright}${colors.cyan}🧠 Installing AgentDB Embedding Dependencies${colors.reset}\n`);

  try {
    // Check if already installed
    try {
      require.resolve('@xenova/transformers');
      console.log(`${colors.yellow}⚠️  @xenova/transformers is already installed${colors.reset}`);
      console.log(`   Checking for updates...`);
    } catch (e) {
      console.log(`${colors.blue}ℹ Installing @xenova/transformers...${colors.reset}`);
    }

    // Determine npm args (avoid shell string interpolation — use array form)
    const npmArgs = options.global
      ? ['install', '-g', '@xenova/transformers']
      : ['install', '@xenova/transformers'];

    console.log(`\n${colors.cyan}📦 Installing optional dependencies:${colors.reset}`);
    console.log(`   - @xenova/transformers (ML models)`);
    console.log(`   - onnxruntime-node (native inference)`);
    console.log('');

    // Install dependencies using spawnSync with args array to prevent shell injection
    try {
      const result = spawnSync('npm', npmArgs, {
        stdio: 'inherit',
        cwd: process.cwd(),
        shell: false
      });
      if (result.status !== 0) {
        throw new Error(`npm exited with code ${result.status ?? 'unknown'}`);
      }

      console.log(`\n${colors.green}✅ Embedding dependencies installed successfully${colors.reset}\n`);

      console.log(`${colors.bright}${colors.magenta}🎉 Next Steps:${colors.reset}`);
      console.log(`   1. Restart your AgentDB instance`);
      console.log(`   2. Real embeddings will be used automatically`);
      console.log(`   3. First run will download model (~90MB): Xenova/all-MiniLM-L6-v2`);
      console.log('');
      console.log(`${colors.cyan}💡 Tip:${colors.reset} Set ${colors.yellow}HUGGINGFACE_API_KEY${colors.reset} for online models`);
      console.log('');

    } catch (installError) {
      console.error(`${colors.red}❌ Installation failed:${colors.reset}`);
      console.error(`   ${(installError as Error).message}`);
      console.log('');
      console.log(`${colors.yellow}Troubleshooting:${colors.reset}`);
      console.log(`   - Ensure you have build tools installed (python3, make, g++)`);
      console.log(`   - On Alpine Linux: apk add --no-cache python3 make g++ gcompat`);
      console.log(`   - On Debian/Ubuntu: apt-get install python3 build-essential`);
      console.log(`   - On macOS: xcode-select --install`);
      process.exit(1);
    }

  } catch (error) {
    console.error(`${colors.red}❌ Command failed:${colors.reset}`);
    console.error(`   ${(error as Error).message}`);
    process.exit(1);
  }
}
