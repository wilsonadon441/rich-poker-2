/**
 * PM2 ecosystem for a rich-poker miner.
 *
 * All runtime identity (wallet, hotkey, port, repo metadata, pm2 name, retrain
 * schedule) comes from the repo's .env file — nothing is hardcoded here.
 *
 * Usage:
 *   pm2 start scripts/miner/ecosystem.config.cjs
 *   pm2 save
 */
const fs = require("fs");
const path = require("path");

const REPO_ROOT = path.resolve(__dirname, "..", "..");

function parseEnvFile(file) {
  const out = {};
  if (!fs.existsSync(file)) return out;
  for (const rawLine of fs.readFileSync(file, "utf8").split("\n")) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const eq = line.indexOf("=");
    if (eq <= 0) continue;
    const key = line.slice(0, eq).trim();
    let value = line.slice(eq + 1).trim();
    if (!value.startsWith('"') && !value.startsWith("'")) {
      const comment = value.indexOf(" #");
      if (comment >= 0) value = value.slice(0, comment).trim();
    }
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    out[key] = value;
  }
  return out;
}

const env = parseEnvFile(path.join(REPO_ROOT, ".env"));
const name = env.POKER44_PM2_NAME || path.basename(REPO_ROOT);
const python = path.join(REPO_ROOT, "miner_env", "bin", "python");

const minerArgs = [
  path.join(REPO_ROOT, "neurons", "miner.py"),
  "--netuid", env.POKER44_NETUID || "126",
  "--wallet.name", env.POKER44_WALLET_NAME || "default",
  "--wallet.hotkey", env.POKER44_WALLET_HOTKEY || "default",
  "--subtensor.network", env.POKER44_NETWORK || "finney",
  "--axon.port", env.POKER44_AXON_PORT || "8091",
  "--logging.debug",
];
const allowlist = (env.POKER44_ALLOWED_VALIDATOR_HOTKEYS || "")
  .split(/\s+/)
  .filter(Boolean);
if (allowlist.length) {
  minerArgs.push("--blacklist.allowed_validator_hotkeys", ...allowlist);
} else {
  minerArgs.push("--blacklist.force_validator_permit");
}

// Pass through every non-empty POKER44_* var. Empty values are dropped so the
// miner's fallbacks still apply (e.g. empty POKER44_MODEL_REPO_COMMIT resolves
// to `git rev-parse HEAD` at startup, keeping the manifest commit accurate).
const passthrough = {};
for (const [key, value] of Object.entries(env)) {
  if (key.startsWith("POKER44_") && value !== "") passthrough[key] = value;
}

module.exports = {
  apps: [
    {
      name,
      cwd: REPO_ROOT,
      script: python,
      args: minerArgs,
      interpreter: "none",
      autorestart: true,
      max_restarts: 50,
      restart_delay: 15000,
      kill_timeout: 10000,
      env: {
        PYTHONPATH: REPO_ROOT,
        ...passthrough,
      },
    },
    {
      // One-shot daily retrain job re-fired by pm2 on the .env cron schedule.
      name: `${name}-retrain`,
      cwd: REPO_ROOT,
      script: path.join(REPO_ROOT, "scripts", "daily_retrain.sh"),
      interpreter: "bash",
      autorestart: false,
      cron_restart: env.POKER44_RETRAIN_CRON || "10 1 * * *",
      env: {
        PYTHONPATH: REPO_ROOT,
      },
    },
  ],
};
