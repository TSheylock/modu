const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
    console.log("Deploying SASOK Platform contracts...");

    // Get the contract factory
    const SasokPlatform = await ethers.getContractFactory("SasokPlatform");
    
    // Deploy the contract
    console.log("Deploying SasokPlatform...");
    const sasokPlatform = await SasokPlatform.deploy();
    await sasokPlatform.deployed();
    
    console.log("SasokPlatform deployed to:", sasokPlatform.address);

    // Save the contract addresses and ABIs
    const contractsDir = path.join(__dirname, "..", "config");
    
    if (!fs.existsSync(contractsDir)) {
        fs.mkdirSync(contractsDir);
    }

    const deploymentInfo = {
        SasokPlatform: {
            address: sasokPlatform.address,
            abi: JSON.parse(sasokPlatform.interface.format('json'))
        }
    };

    fs.writeFileSync(
        path.join(contractsDir, "contracts.json"),
        JSON.stringify(deploymentInfo, null, 2)
    );

    console.log("Deployment information saved to:", path.join(contractsDir, "contracts.json"));

    // Verify contract on Etherscan
    if (process.env.ETHERSCAN_API_KEY) {
        console.log("Verifying contract on Etherscan...");
        await run("verify:verify", {
            address: sasokPlatform.address,
            constructorArguments: []
        });
        console.log("Contract verified on Etherscan");
    }
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });
