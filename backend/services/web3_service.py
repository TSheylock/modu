from web3 import Web3
from eth_account import Account
import json
import os
from typing import Dict, List, Optional
import logging
from datetime import datetime

class Web3Service:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.w3 = None
        self.contract = None
        self.initialize_web3()

    def initialize_web3(self):
        """Initialize Web3 connection and contract"""
        try:
            # Connect to network
            provider_url = self.config['web3']['provider_url']
            self.w3 = Web3(Web3.HTTPProvider(provider_url))
            
            # Load contract ABI and address
            contract_path = os.path.join(
                os.path.dirname(__file__),
                '../config/contracts.json'
            )
            
            with open(contract_path, 'r') as f:
                contract_data = json.load(f)
            
            self.contract = self.w3.eth.contract(
                address=contract_data['SasokPlatform']['address'],
                abi=contract_data['SasokPlatform']['abi']
            )
            
            self.logger.info("Web3 service initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing Web3 service: {str(e)}")
            raise

    async def register_user(self, wallet_address: str) -> Dict:
        """Register a new user on the platform"""
        try:
            # Build transaction
            nonce = self.w3.eth.get_transaction_count(wallet_address)
            tx = self.contract.functions.registerUser().build_transaction({
                'from': wallet_address,
                'nonce': nonce,
                'gas': 200000,
                'gasPrice': self.w3.eth.gas_price
            })
            
            return {
                'success': True,
                'transaction': tx
            }
        except Exception as e:
            self.logger.error(f"Error registering user: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    async def record_interaction(self, 
                               wallet_address: str,
                               interaction_type: str,
                               metadata: str) -> Dict:
        """Record a user interaction on the blockchain"""
        try:
            # Build transaction
            nonce = self.w3.eth.get_transaction_count(wallet_address)
            tx = self.contract.functions.recordInteraction(
                interaction_type,
                metadata
            ).build_transaction({
                'from': wallet_address,
                'nonce': nonce,
                'gas': 200000,
                'gasPrice': self.w3.eth.gas_price
            })
            
            return {
                'success': True,
                'transaction': tx
            }
        except Exception as e:
            self.logger.error(f"Error recording interaction: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    async def mint_nft(self, 
                      recipient: str,
                      token_uri: str,
                      admin_wallet: str) -> Dict:
        """Mint a new NFT for a user"""
        try:
            # Build transaction
            nonce = self.w3.eth.get_transaction_count(admin_wallet)
            tx = self.contract.functions.mintNFT(
                recipient,
                token_uri
            ).build_transaction({
                'from': admin_wallet,
                'nonce': nonce,
                'gas': 300000,
                'gasPrice': self.w3.eth.gas_price
            })
            
            return {
                'success': True,
                'transaction': tx
            }
        except Exception as e:
            self.logger.error(f"Error minting NFT: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    async def get_user_profile(self, wallet_address: str) -> Dict:
        """Get user profile from the contract"""
        try:
            profile = await self.contract.functions.getUserProfile(
                wallet_address
            ).call()
            
            return {
                'success': True,
                'profile': {
                    'isRegistered': profile[0],
                    'reputation': profile[1],
                    'interactionCount': profile[2],
                    'ownedTokens': profile[3],
                    'lastInteraction': profile[4]
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting user profile: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    async def get_interaction(self, interaction_id: int) -> Dict:
        """Get interaction details from the contract"""
        try:
            interaction = await self.contract.functions.getInteraction(
                interaction_id
            ).call()
            
            return {
                'success': True,
                'interaction': {
                    'user': interaction[0],
                    'interactionType': interaction[1],
                    'metadata': interaction[2],
                    'timestamp': interaction[3],
                    'verified': interaction[4]
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting interaction: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    async def verify_interaction(self, 
                               interaction_id: int,
                               admin_wallet: str) -> Dict:
        """Verify an interaction"""
        try:
            nonce = self.w3.eth.get_transaction_count(admin_wallet)
            tx = self.contract.functions.verifyInteraction(
                interaction_id
            ).build_transaction({
                'from': admin_wallet,
                'nonce': nonce,
                'gas': 100000,
                'gasPrice': self.w3.eth.gas_price
            })
            
            return {
                'success': True,
                'transaction': tx
            }
        except Exception as e:
            self.logger.error(f"Error verifying interaction: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    async def update_user_reputation(self,
                                   user_address: str,
                                   reputation: int,
                                   admin_wallet: str) -> Dict:
        """Update user reputation"""
        try:
            nonce = self.w3.eth.get_transaction_count(admin_wallet)
            tx = self.contract.functions.updateUserProfile(
                user_address,
                reputation
            ).build_transaction({
                'from': admin_wallet,
                'nonce': nonce,
                'gas': 100000,
                'gasPrice': self.w3.eth.gas_price
            })
            
            return {
                'success': True,
                'transaction': tx
            }
        except Exception as e:
            self.logger.error(f"Error updating reputation: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_contract_address(self) -> str:
        """Get the deployed contract address"""
        return self.contract.address

    def is_connected(self) -> bool:
        """Check if Web3 is connected"""
        return self.w3.is_connected()

    async def get_network_info(self) -> Dict:
        """Get current network information"""
        try:
            return {
                'success': True,
                'network': {
                    'chainId': self.w3.eth.chain_id,
                    'blockNumber': self.w3.eth.block_number,
                    'gasPrice': self.w3.eth.gas_price,
                    'isConnected': self.w3.is_connected()
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting network info: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
