import { HardhatUserConfig } from "hardhat/config";
import "@nomicfoundation/hardhat-toolbox";
import * as dotenv from "dotenv";

dotenv.config();

const PRIVATE_KEY = process.env.PRIVATE_KEY || "0000000000000000000000000000000000000000000000000000000000000000";
const SASKE_MAINNET_RPC_URL = process.env.SASKE_MAINNET_RPC_URL || "https://mainnet.saske.xyz";
const SASKE_TESTNET_RPC_URL = process.env.SASKE_TESTNET_RPC_URL || "https://testnet.saske.xyz";

const config: HardhatUserConfig = {
  solidity: {
    version: "0.8.19",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200
      }
    }
  },
  networks: {
    hardhat: {
      chainId: 31337
    },
    saskeMainnet: {
      url: SASKE_MAINNET_RPC_URL,
      accounts: [PRIVATE_KEY],
      chainId: 1337
    },
    saskeTestnet: {
      url: SASKE_TESTNET_RPC_URL,
      accounts: [PRIVATE_KEY],
      chainId: 1338
    }
  },
  paths: {
    sources: "./src/contracts",
    tests: "./test",
    cache: "./cache",
    artifacts: "./artifacts"
  },
  mocha: {
    timeout: 40000
  }
};

export default config; 