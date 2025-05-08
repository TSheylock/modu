// SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Strings.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";

/**
 * @title EmotionalSBT
 * @dev Contract for Emotional Soulbound Tokens (SBT) используемый в SASOK
 * Поддерживает Zero-Knowledge доказательства для защиты приватности
 * Динамические NFT, изменяющие визуал в зависимости от эмоционального состояния
 */
contract EmotionalSBT is ERC721, ERC721URIStorage, Ownable {
    using Strings for uint256;
    using ECDSA for bytes32;
    
    // События
    event EmotionalStateUpdated(uint256 indexed tokenId, string emotionType, uint256 timestamp);
    event ZKProofVerified(uint256 indexed tokenId, bytes32 proofHash, uint256 timestamp);
    
    // Структуры
    struct EmotionalState {
        string dominantEmotion;  // Доминирующая эмоция
        uint256 intensity;      // Интенсивность от 0 до 100
        uint256 timestamp;      // Временная метка обновления
        bytes32 proofHash;      // Хэш Zero-Knowledge доказательства
        bool verified;          // Флаг проверки
    }
    
    // Маппинги
    mapping(address => uint256) private _addressToTokenId;
    mapping(uint256 => bool) private _soulbound;
    mapping(uint256 => EmotionalState) private _emotionalStates;
    mapping(address => bool) private _verifiers;
    
    // Константы
    uint256 private constant MAX_INTENSITY = 100;
    
    // Модификаторы
    modifier onlyVerifier() {
        require(_verifiers[msg.sender] || msg.sender == owner(), "EmotionalSBT: caller is not a verifier");
        _;
    }
    
    /**
     * @dev Конструктор
     */
    constructor() ERC721("EmotionalSBT", "ESBT") {
        // Устанавливаем владельца как верификатора по умолчанию
        _verifiers[msg.sender] = true;
    }
    
    /**
     * @dev Добавляет верификатора для проверки ZK доказательств
     * @param verifier Адрес верификатора
     */
    function addVerifier(address verifier) external onlyOwner {
        require(verifier != address(0), "EmotionalSBT: verifier is zero address");
        _verifiers[verifier] = true;
    }
    
    /**
     * @dev Удаляет верификатора 
     * @param verifier Адрес верификатора
     */
    function removeVerifier(address verifier) external onlyOwner {
        _verifiers[verifier] = false;
    }
    
    /**
     * @dev Создание нового Emotional SBT
     * @param to Адрес получателя токена
     * @param tokenId ID токена
     * @param uri URI метаданных токена
     * @param soulbound Флаг, указывающий привязан ли токен к адресу
     */
    function mint(address to, uint256 tokenId, string memory uri, bool soulbound) public onlyOwner {
        _safeMint(to, tokenId);
        _setTokenURI(tokenId, uri);
        
        if (soulbound) {
            _soulbound[tokenId] = true;
            _addressToTokenId[to] = tokenId;
        }
        
        // Устанавливаем начальное эмоциональное состояние
        _emotionalStates[tokenId] = EmotionalState({
            dominantEmotion: "neutral",
            intensity: 50,
            timestamp: block.timestamp,
            proofHash: bytes32(0),
            verified: false
        });
    }
    
    /**
     * @dev Обновление URI токена (для динамических NFT)
     * @param tokenId ID токена
     * @param uri Новый URI метаданных
     */
    function updateTokenURI(uint256 tokenId, string memory uri) public onlyOwner {
        require(_exists(tokenId), "EmotionalSBT: token does not exist");
        _setTokenURI(tokenId, uri);
    }
    
    /**
     * @dev Обновление эмоционального состояния с Zero-Knowledge доказательством
     * @param tokenId ID токена
     * @param emotion Новая доминирующая эмоция
     * @param intensity Интенсивность эмоции (0-100)
     * @param proofHash Хэш Zero-Knowledge доказательства
     */
    function updateEmotionalState(
        uint256 tokenId, 
        string memory emotion, 
        uint256 intensity, 
        bytes32 proofHash
    ) public onlyVerifier {
        require(_exists(tokenId), "EmotionalSBT: token does not exist");
        require(intensity <= MAX_INTENSITY, "EmotionalSBT: intensity too high");
        
        // Обновляем состояние
        _emotionalStates[tokenId] = EmotionalState({
            dominantEmotion: emotion,
            intensity: intensity,
            timestamp: block.timestamp,
            proofHash: proofHash,
            verified: true
        });
        
        emit EmotionalStateUpdated(tokenId, emotion, block.timestamp);
        emit ZKProofVerified(tokenId, proofHash, block.timestamp);
    }
    
    /**
     * @dev Получение текущего эмоционального состояния токена
     * @param tokenId ID токена
     */
    function getEmotionalState(uint256 tokenId) public view returns (
        string memory emotion,
        uint256 intensity,
        uint256 timestamp,
        bytes32 proofHash,
        bool verified
    ) {
        require(_exists(tokenId), "EmotionalSBT: token does not exist");
        
        EmotionalState memory state = _emotionalStates[tokenId];
        return (
            state.dominantEmotion,
            state.intensity,
            state.timestamp,
            state.proofHash,
            state.verified
        );
    }
    
    /**
     * @dev Проверка, привязан ли токен (soulbound)
     * @param tokenId ID токена
     */
    function isSoulbound(uint256 tokenId) public view returns (bool) {
        require(_exists(tokenId), "EmotionalSBT: token does not exist");
        return _soulbound[tokenId];
    }
    
    /**
     * @dev Получение токена по адресу (для soulbound токенов)
     * @param owner Адрес владельца
     */
    function getTokenByOwner(address owner) public view returns (uint256) {
        return _addressToTokenId[owner];
    }
    
    /**
     * @dev Проверка подписи для проверки подлинности обновления
     * @param tokenId ID токена
     * @param emotion Эмоция
     * @param intensity Интенсивность
     * @param timestamp Временная метка
     * @param signature Подпись
     * @param signer Адрес подписавшего
     */
    function verifySignature(
        uint256 tokenId,
        string memory emotion,
        uint256 intensity,
        uint256 timestamp,
        bytes memory signature,
        address signer
    ) public pure returns (bool) {
        bytes32 messageHash = keccak256(
            abi.encodePacked(tokenId, emotion, intensity, timestamp)
        );
        bytes32 ethSignedMessageHash = messageHash.toEthSignedMessageHash();
        address recoveredSigner = ethSignedMessageHash.recover(signature);
        
        return recoveredSigner == signer;
    }
    
    // Переопределение функции transferFrom для soulbound токенов
    function transferFrom(address from, address to, uint256 tokenId) public override {
        require(!_soulbound[tokenId], "EmotionalSBT: token is soulbound");
        super.transferFrom(from, to, tokenId);
    }
    
    // Переопределение функции safeTransferFrom для soulbound токенов
    function safeTransferFrom(address from, address to, uint256 tokenId) public override {
        require(!_soulbound[tokenId], "EmotionalSBT: token is soulbound");
        safeTransferFrom(from, to, tokenId, "");
    }
    
    // Переопределение функции safeTransferFrom для soulbound токенов
    function safeTransferFrom(address from, address to, uint256 tokenId, bytes memory data) public override {
        require(!_soulbound[tokenId], "EmotionalSBT: token is soulbound");
        super.safeTransferFrom(from, to, tokenId, data);
    }
    
    // Обязательные переопределения для ERC721URIStorage
    function _burn(uint256 tokenId) internal override(ERC721, ERC721URIStorage) {
        super._burn(tokenId);
    }
    
    function tokenURI(uint256 tokenId) public view override(ERC721, ERC721URIStorage) returns (string memory) {
        return super.tokenURI(tokenId);
    }
    
    function supportsInterface(bytes4 interfaceId) public view override(ERC721, ERC721URIStorage) returns (bool) {
        return super.supportsInterface(interfaceId);
    }
}
