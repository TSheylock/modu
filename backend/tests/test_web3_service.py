import pytest
from ..services.web3_service import Web3Service
from web3 import Web3
import json
import os
from eth_account import Account
import eth_tester
from eth_utils import to_checksum_address

@pytest.fixture
def config():
    return {
        'web3': {
            'provider_url': 'http://localhost:8545',  # Local Hardhat node
            'chain_id': 1337
        }
    }

@pytest.fixture
def web3_service(config):
    return Web3Service(config)

@pytest.fixture
def accounts():
    # Create test accounts
    return [Account.create() for _ in range(3)]

class TestWeb3Service:
    @pytest.mark.asyncio
    async def test_initialization(self, web3_service):
        assert web3_service.w3 is not None
        assert web3_service.contract is not None
        assert web3_service.is_connected() is True

    @pytest.mark.asyncio
    async def test_register_user(self, web3_service, accounts):
        # Get transaction for user registration
        result = await web3_service.register_user(accounts[0].address)
        assert result['success'] is True
        assert 'transaction' in result
        
        tx = result['transaction']
        assert tx['from'] == accounts[0].address
        assert 'gas' in tx
        assert 'gasPrice' in tx

    @pytest.mark.asyncio
    async def test_record_interaction(self, web3_service, accounts):
        # First register user
        await web3_service.register_user(accounts[0].address)
        
        # Record interaction
        result = await web3_service.record_interaction(
            accounts[0].address,
            "TEST_INTERACTION",
            "Test metadata"
        )
        
        assert result['success'] is True
        assert 'transaction' in result
        
        tx = result['transaction']
        assert tx['from'] == accounts[0].address
        assert 'gas' in tx
        assert 'gasPrice' in tx

    @pytest.mark.asyncio
    async def test_mint_nft(self, web3_service, accounts):
        admin_wallet = accounts[0].address
        recipient = accounts[1].address
        token_uri = "ipfs://test-uri"

        # Register recipient
        await web3_service.register_user(recipient)
        
        # Mint NFT
        result = await web3_service.mint_nft(
            recipient,
            token_uri,
            admin_wallet
        )
        
        assert result['success'] is True
        assert 'transaction' in result
        
        tx = result['transaction']
        assert tx['from'] == admin_wallet
        assert 'gas' in tx
        assert 'gasPrice' in tx

    @pytest.mark.asyncio
    async def test_get_user_profile(self, web3_service, accounts):
        user_address = accounts[0].address
        
        # Register user first
        await web3_service.register_user(user_address)
        
        # Get profile
        result = await web3_service.get_user_profile(user_address)
        
        assert result['success'] is True
        assert 'profile' in result
        
        profile = result['profile']
        assert profile['isRegistered'] is True
        assert 'reputation' in profile
        assert 'interactionCount' in profile
        assert 'ownedTokens' in profile
        assert 'lastInteraction' in profile

    @pytest.mark.asyncio
    async def test_verify_interaction(self, web3_service, accounts):
        user_address = accounts[0].address
        admin_wallet = accounts[1].address
        
        # Register user
        await web3_service.register_user(user_address)
        
        # Record interaction
        interaction_result = await web3_service.record_interaction(
            user_address,
            "TEST_INTERACTION",
            "Test metadata"
        )
        
        # Get interaction ID from event logs
        tx_receipt = await web3_service.w3.eth.wait_for_transaction_receipt(
            interaction_result['transaction']['hash']
        )
        interaction_id = 1  # First interaction ID
        
        # Verify interaction
        result = await web3_service.verify_interaction(
            interaction_id,
            admin_wallet
        )
        
        assert result['success'] is True
        assert 'transaction' in result

    @pytest.mark.asyncio
    async def test_update_user_reputation(self, web3_service, accounts):
        user_address = accounts[0].address
        admin_wallet = accounts[1].address
        new_reputation = 100
        
        # Register user
        await web3_service.register_user(user_address)
        
        # Update reputation
        result = await web3_service.update_user_reputation(
            user_address,
            new_reputation,
            admin_wallet
        )
        
        assert result['success'] is True
        assert 'transaction' in result

    @pytest.mark.asyncio
    async def test_get_network_info(self, web3_service):
        result = await web3_service.get_network_info()
        
        assert result['success'] is True
        assert 'network' in result
        
        network = result['network']
        assert 'chainId' in network
        assert 'blockNumber' in network
        assert 'gasPrice' in network
        assert 'isConnected' in network
        assert network['isConnected'] is True

    @pytest.mark.asyncio
    async def test_complete_user_flow(self, web3_service, accounts):
        user_address = accounts[0].address
        admin_wallet = accounts[1].address
        
        # 1. Register user
        reg_result = await web3_service.register_user(user_address)
        assert reg_result['success'] is True
        
        # 2. Record interaction
        int_result = await web3_service.record_interaction(
            user_address,
            "COMPLETE_TASK",
            "Completed test task"
        )
        assert int_result['success'] is True
        
        # 3. Verify interaction
        ver_result = await web3_service.verify_interaction(1, admin_wallet)
        assert ver_result['success'] is True
        
        # 4. Update reputation
        rep_result = await web3_service.update_user_reputation(
            user_address,
            50,
            admin_wallet
        )
        assert rep_result['success'] is True
        
        # 5. Mint NFT reward
        nft_result = await web3_service.mint_nft(
            user_address,
            "ipfs://reward-nft",
            admin_wallet
        )
        assert nft_result['success'] is True
        
        # 6. Verify final state
        profile_result = await web3_service.get_user_profile(user_address)
        assert profile_result['success'] is True
        
        profile = profile_result['profile']
        assert profile['isRegistered'] is True
        assert profile['reputation'] == 50
        assert profile['interactionCount'] == 1
        assert len(profile['ownedTokens']) == 1
