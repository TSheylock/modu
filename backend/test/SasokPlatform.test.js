const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("SasokPlatform", function () {
    let SasokPlatform;
    let sasokPlatform;
    let owner;
    let user1;
    let user2;

    beforeEach(async function () {
        // Get contract factory and signers
        SasokPlatform = await ethers.getContractFactory("SasokPlatform");
        [owner, user1, user2] = await ethers.getSigners();

        // Deploy contract
        sasokPlatform = await SasokPlatform.deploy();
        await sasokPlatform.deployed();
    });

    describe("Deployment", function () {
        it("Should set the right owner", async function () {
            expect(await sasokPlatform.owner()).to.equal(owner.address);
        });

        it("Should have correct name and symbol", async function () {
            expect(await sasokPlatform.name()).to.equal("SASOK Platform Token");
            expect(await sasokPlatform.symbol()).to.equal("SASOK");
        });
    });

    describe("User Registration", function () {
        it("Should allow users to register", async function () {
            await sasokPlatform.connect(user1).registerUser();
            const userProfile = await sasokPlatform.getUserProfile(user1.address);
            expect(userProfile.isRegistered).to.be.true;
        });

        it("Should not allow double registration", async function () {
            await sasokPlatform.connect(user1).registerUser();
            await expect(
                sasokPlatform.connect(user1).registerUser()
            ).to.be.revertedWith("User already registered");
        });
    });

    describe("Interactions", function () {
        beforeEach(async function () {
            await sasokPlatform.connect(user1).registerUser();
        });

        it("Should record interactions", async function () {
            const tx = await sasokPlatform.connect(user1).recordInteraction(
                "TEST",
                "Test metadata"
            );
            const receipt = await tx.wait();
            const event = receipt.events.find(e => e.event === 'InteractionRecorded');
            expect(event).to.not.be.undefined;
        });

        it("Should not allow unregistered users to record interactions", async function () {
            await expect(
                sasokPlatform.connect(user2).recordInteraction("TEST", "Test metadata")
            ).to.be.revertedWith("User not registered");
        });

        it("Should allow owner to verify interactions", async function () {
            const tx = await sasokPlatform.connect(user1).recordInteraction(
                "TEST",
                "Test metadata"
            );
            const receipt = await tx.wait();
            const event = receipt.events.find(e => e.event === 'InteractionRecorded');
            const interactionId = event.args.interactionId;

            await sasokPlatform.verifyInteraction(interactionId);
            const interaction = await sasokPlatform.getInteraction(interactionId);
            expect(interaction.verified).to.be.true;
        });
    });

    describe("NFT Functionality", function () {
        beforeEach(async function () {
            await sasokPlatform.connect(user1).registerUser();
        });

        it("Should allow owner to mint NFTs", async function () {
            const tokenURI = "ipfs://test-uri";
            const tx = await sasokPlatform.mintNFT(user1.address, tokenURI);
            const receipt = await tx.wait();
            const event = receipt.events.find(e => e.event === 'NFTMinted');
            expect(event).to.not.be.undefined;

            const tokenId = event.args.tokenId;
            expect(await sasokPlatform.ownerOf(tokenId)).to.equal(user1.address);
            expect(await sasokPlatform.tokenURI(tokenId)).to.equal(tokenURI);
        });

        it("Should not allow minting to unregistered users", async function () {
            await expect(
                sasokPlatform.mintNFT(user2.address, "ipfs://test-uri")
            ).to.be.revertedWith("Recipient not registered");
        });
    });

    describe("Reputation System", function () {
        beforeEach(async function () {
            await sasokPlatform.connect(user1).registerUser();
        });

        it("Should allow owner to update user reputation", async function () {
            const newReputation = 100;
            await sasokPlatform.updateUserProfile(user1.address, newReputation);
            const userProfile = await sasokPlatform.getUserProfile(user1.address);
            expect(userProfile.reputation).to.equal(newReputation);
        });

        it("Should not allow non-owners to update reputation", async function () {
            await expect(
                sasokPlatform.connect(user1).updateUserProfile(user1.address, 100)
            ).to.be.revertedWith("Ownable: caller is not the owner");
        });
    });

    describe("Platform Management", function () {
        it("Should allow owner to pause and unpause", async function () {
            await sasokPlatform.pause();
            expect(await sasokPlatform.paused()).to.be.true;

            await sasokPlatform.unpause();
            expect(await sasokPlatform.paused()).to.be.false;
        });

        it("Should not allow non-owners to pause", async function () {
            await expect(
                sasokPlatform.connect(user1).pause()
            ).to.be.revertedWith("Ownable: caller is not the owner");
        });
    });

    describe("Integration Tests", function () {
        it("Should handle complete user journey", async function () {
            // Register user
            await sasokPlatform.connect(user1).registerUser();
            
            // Record interaction
            const tx1 = await sasokPlatform.connect(user1).recordInteraction(
                "COMPLETE_TASK",
                "Completed first task"
            );
            const receipt1 = await tx1.wait();
            const interactionId = receipt1.events.find(
                e => e.event === 'InteractionRecorded'
            ).args.interactionId;
            
            // Verify interaction
            await sasokPlatform.verifyInteraction(interactionId);
            
            // Update reputation
            await sasokPlatform.updateUserProfile(user1.address, 50);
            
            // Mint NFT reward
            const tx2 = await sasokPlatform.mintNFT(
                user1.address,
                "ipfs://reward-nft"
            );
            const receipt2 = await tx2.wait();
            const tokenId = receipt2.events.find(
                e => e.event === 'NFTMinted'
            ).args.tokenId;
            
            // Verify final state
            const userProfile = await sasokPlatform.getUserProfile(user1.address);
            expect(userProfile.isRegistered).to.be.true;
            expect(userProfile.reputation).to.equal(50);
            expect(userProfile.interactionCount).to.equal(1);
            expect(await sasokPlatform.ownerOf(tokenId)).to.equal(user1.address);
        });
    });
});
