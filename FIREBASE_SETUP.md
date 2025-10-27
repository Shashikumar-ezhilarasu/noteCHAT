# Firebase Setup Instructions

## Your Firebase Project Details

- **Project ID**: `notechat-26c38`
- **Storage Bucket**: `notechat-26c38.firebasestorage.app`
- **Web App**: Already configured in frontend

## Get Service Account Credentials

1. **Go to Firebase Console**:

   ```
   https://console.firebase.google.com/project/notechat-26c38/settings/serviceaccounts/adminsdk
   ```

2. **Generate New Private Key**:

   - Click "Generate new private key"
   - Save the downloaded JSON file as `firebase_admin_config.json` in the `backend/` folder

3. **File Structure Check**:
   ```
   backend/
   ├── firebase_admin_config.json  ← Your actual credentials
   ├── upload_documents.py
   ├── main.py
   └── ...
   ```

## Quick Test

Run this to verify your setup:

```bash
cd backend
python3 -c "
import json
with open('firebase_admin_config.json') as f:
    config = json.load(f)
    if 'your-private-key-id' in config.get('private_key_id', ''):
        print('❌ Still using template - please replace with real credentials')
    else:
        print('✅ Firebase credentials configured!')
        print(f'Project: {config.get(\"project_id\", \"unknown\")}')
"
```

## Upload Your Documents

```bash
# After adding credentials
./upload_and_start.sh
```

This will:

1. Upload all files from `../NOTES/` to Firebase Storage
2. Start the backend (downloads files and processes them)
3. Start the frontend
4. Open http://localhost:3000 to use your AI assistant
