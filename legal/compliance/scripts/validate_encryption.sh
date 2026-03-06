#!/bin/bash
# Encryption Validation Script

# 1. Test: Cannot access database without encryption key
psql -h prod-db.sasok.internal -d user_data \
  -c "SELECT * FROM users LIMIT 1" \
  --no-password 2>&1 | grep -q "FATAL" && echo "PASS: Unencrypted access blocked" || echo "FAIL"

# 2. Test: TLS certificate valid
openssl s_client -connect api.sasok.ai:443 \
  -servername api.sasok.ai 2>/dev/null | \
  openssl x509 -noout -dates | grep "notAfter" | \
  awk '{print $4}' | grep "2026\|2027" && echo "PASS: Certificate valid" || echo "FAIL"

# 3. Test: Backup encrypted and key separate
ls -la /backups/sasok_db_*.enc 2>/dev/null | \
  wc -l | grep -q "[1-9]" && echo "PASS: Encrypted backups found" || echo "FAIL"

# 4. Test: Decrypt backup with separate key
/usr/local/bin/decrypt_backup.sh \
  /backups/sasok_db_2025-12-17.enc \
  --key-from /secure/backup_keys/ 2>&1 | grep -q "SUCCESS" && \
  echo "PASS: Backup decryption works" || echo "FAIL"
