from web3 import Web3
from eth_account.messages import encode_defunct
from typing import Dict, List, Optional
import json
import logging
from datetime import datetime

class Web3Handler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Initialize Web3 with Infura (replace with your project ID in production)
        self.w3 = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/YOUR-PROJECT-ID'))
        self.connected_wallets = {}
        self.contract_cache = {}

    async def connect_wallet(self, wallet_address: str) -> Dict:
        """
        Connect and validate a wallet address
        """
        try:
            if not self.w3.is_address(wallet_address):
                raise ValueError("Invalid Ethereum address")

            wallet_data = {
                "address": wallet_address,
                "network": "ethereum",
                "connected_at": datetime.utcnow().isoformat(),
                "balance": self.w3.eth.get_balance(wallet_address)
            }

            self.connected_wallets[wallet_address] = wallet_data
            return wallet_data

        except Exception as e:
            self.logger.error(f"Error connecting wallet: {str(e)}")
            raise

    async def verify_signature(self, message: str, signature: str, address: str) -> bool:
        """
        Verify that a message was signed by the specified address
        """
        try:
            message_hash = encode_defunct(text=message)
            recovered_address = self.w3.eth.account.recover_message(message_hash, signature=signature)
            return recovered_address.lower() == address.lower()

        except Exception as e:
            self.logger.error(f"Error verifying signature: {str(e)}")
            return False

    async def get_transaction_history(self, address: str, limit: int = 10) -> List[Dict]:
        """
        Get transaction history for an address
        """
        try:
            transactions = []
            latest_block = self.w3.eth.block_number

            for i in range(limit):
                block = self.w3.eth.get_block(latest_block - i, full_transactions=True)
                for tx in block.transactions:
                    if tx['to'] and (tx['to'].lower() == address.lower() or 
                                   tx['from'].lower() == address.lower()):
                        transactions.append({
                            "hash": tx['hash'].hex(),
                            "from": tx['from'],
                            "to": tx['to'],
                            "value": self.w3.from_wei(tx['value'], 'ether'),
                            "block": tx['blockNumber'],
                            "timestamp": block.timestamp
                        })

            return transactions

        except Exception as e:
            self.logger.error(f"Error getting transaction history: {str(e)}")
            raise

    async def interact_with_contract(self, 
                                   contract_address: str, 
                                   abi_path: str, 
                                   method_name: str, 
                                   params: List = None) -> Dict:
        """
        Interact with a smart contract
        """
        try:
            # Load contract ABI
            if contract_address not in self.contract_cache:
                with open(abi_path, 'r') as f:
                    contract_abi = json.load(f)
                contract = self.w3.eth.contract(
                    address=self.w3.to_checksum_address(contract_address),
                    abi=contract_abi
                )
                self.contract_cache[contract_address] = contract
            else:
                contract = self.contract_cache[contract_address]

            # Call contract method
            method = getattr(contract.functions, method_name)
            if params:
                result = method(*params).call()
            else:
                result = method().call()

            return {
                "success": True,
                "result": result,
                "contract": contract_address,
                "method": method_name
            }

        except Exception as e:
            self.logger.error(f"Error interacting with contract: {str(e)}")
            raise

    async def get_nft_data(self, contract_address: str, token_id: int) -> Dict:
        """
        Get NFT metadata from a contract
        """
        try:
            # ERC721 standard interface
            erc721_abi = [
                {
                    "inputs": [{"internalType": "uint256", "name": "tokenId", "type": "uint256"}],
                    "name": "tokenURI",
                    "outputs": [{"internalType": "string", "name": "", "type": "string"}],
                    "stateMutability": "view",
                    "type": "function"
                }
            ]

            contract = self.w3.eth.contract(
                address=self.w3.to_checksum_address(contract_address),
                abi=erc721_abi
            )

            token_uri = contract.functions.tokenURI(token_id).call()

            return {
                "token_id": token_id,
                "contract": contract_address,
                "token_uri": token_uri
            }

        except Exception as e:
            self.logger.error(f"Error getting NFT data: {str(e)}")
            raise

    async def get_wallet_stats(self, address: str) -> Dict:
        """
        Get comprehensive wallet statistics
        """
        try:
            balance = self.w3.eth.get_balance(address)
            transaction_count = self.w3.eth.get_transaction_count(address)

            return {
                "address": address,
                "balance_eth": self.w3.from_wei(balance, 'ether'),
                "transaction_count": transaction_count,
                "last_updated": datetime.utcnow().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error getting wallet stats: {str(e)}")
            raise

    def disconnect_wallet(self, wallet_address: str) -> bool:
        """
        Disconnect a wallet
        """
        try:
            if wallet_address in self.connected_wallets:
                del self.connected_wallets[wallet_address]
                return True
            return False

        except Exception as e:
            self.logger.error(f"Error disconnecting wallet: {str(e)}")
            return False
