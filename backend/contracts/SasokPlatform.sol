// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

/**
 * @title SasokPlatform
 * @dev Main contract for the SASOK platform, handling user interactions and NFT functionality
 */
contract SasokPlatform is ERC721URIStorage, Pausable, Ownable {
    using Counters for Counters.Counter;
    Counters.Counter private _tokenIds;
    
    // Mapping for user profiles
    mapping(address => UserProfile) public userProfiles;
    
    // Mapping for interaction records
    mapping(uint256 => Interaction) public interactions;
    
    // Counter for interaction IDs
    Counters.Counter private _interactionIds;
    
    // Structs
    struct UserProfile {
        bool isRegistered;
        uint256 reputation;
        uint256 interactionCount;
        uint256[] ownedTokens;
        uint256 lastInteraction;
    }
    
    struct Interaction {
        address user;
        string interactionType;
        string metadata;
        uint256 timestamp;
        bool verified;
    }
    
    // Events
    event UserRegistered(address indexed user, uint256 timestamp);
    event InteractionRecorded(uint256 indexed interactionId, address indexed user, string interactionType);
    event ReputationUpdated(address indexed user, uint256 newReputation);
    event NFTMinted(address indexed to, uint256 indexed tokenId, string uri);
    
    // Constructor
    constructor() ERC721("SASOK Platform Token", "SASOK") {}
    
    // Modifiers
    modifier onlyRegistered() {
        require(userProfiles[msg.sender].isRegistered, "User not registered");
        _;
    }
    
    // User Management Functions
    function registerUser() external {
        require(!userProfiles[msg.sender].isRegistered, "User already registered");
        
        userProfiles[msg.sender] = UserProfile({
            isRegistered: true,
            reputation: 0,
            interactionCount: 0,
            ownedTokens: new uint256[](0),
            lastInteraction: block.timestamp
        });
        
        emit UserRegistered(msg.sender, block.timestamp);
    }
    
    function updateUserProfile(address user, uint256 reputation) external onlyOwner {
        require(userProfiles[user].isRegistered, "User not registered");
        
        userProfiles[user].reputation = reputation;
        emit ReputationUpdated(user, reputation);
    }
    
    // Interaction Functions
    function recordInteraction(
        string memory interactionType,
        string memory metadata
    ) external onlyRegistered returns (uint256) {
        _interactionIds.increment();
        uint256 newInteractionId = _interactionIds.current();
        
        interactions[newInteractionId] = Interaction({
            user: msg.sender,
            interactionType: interactionType,
            metadata: metadata,
            timestamp: block.timestamp,
            verified: false
        });
        
        userProfiles[msg.sender].interactionCount++;
        userProfiles[msg.sender].lastInteraction = block.timestamp;
        
        emit InteractionRecorded(newInteractionId, msg.sender, interactionType);
        return newInteractionId;
    }
    
    function verifyInteraction(uint256 interactionId) external onlyOwner {
        require(interactions[interactionId].timestamp > 0, "Interaction does not exist");
        interactions[interactionId].verified = true;
    }
    
    // NFT Functions
    function mintNFT(
        address recipient,
        string memory tokenURI
    ) external onlyOwner returns (uint256) {
        require(userProfiles[recipient].isRegistered, "Recipient not registered");
        
        _tokenIds.increment();
        uint256 newTokenId = _tokenIds.current();
        
        _mint(recipient, newTokenId);
        _setTokenURI(newTokenId, tokenURI);
        
        userProfiles[recipient].ownedTokens.push(newTokenId);
        
        emit NFTMinted(recipient, newTokenId, tokenURI);
        return newTokenId;
    }
    
    // View Functions
    function getUserProfile(address user) external view returns (
        bool isRegistered,
        uint256 reputation,
        uint256 interactionCount,
        uint256[] memory ownedTokens,
        uint256 lastInteraction
    ) {
        UserProfile memory profile = userProfiles[user];
        return (
            profile.isRegistered,
            profile.reputation,
            profile.interactionCount,
            profile.ownedTokens,
            profile.lastInteraction
        );
    }
    
    function getInteraction(uint256 interactionId) external view returns (
        address user,
        string memory interactionType,
        string memory metadata,
        uint256 timestamp,
        bool verified
    ) {
        Interaction memory interaction = interactions[interactionId];
        return (
            interaction.user,
            interaction.interactionType,
            interaction.metadata,
            interaction.timestamp,
            interaction.verified
        );
    }
    
    // Platform Management Functions
    function pause() external onlyOwner {
        _pause();
    }
    
    function unpause() external onlyOwner {
        _unpause();
    }
    
    // Override required functions
    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 tokenId,
        uint256 batchSize
    ) internal override whenNotPaused {
        super._beforeTokenTransfer(from, to, tokenId, batchSize);
    }
    
    // The following functions are overrides required by Solidity
    function supportsInterface(bytes4 interfaceId)
        public
        view
        override(ERC721URIStorage)
        returns (bool)
    {
        return super.supportsInterface(interfaceId);
    }
}
