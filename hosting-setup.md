# Hosting Setup Guide

## ✅ Your Application is Now Cloud-Ready!

### Changes Made:
- **Vector Storage**: Moved from local ChromaDB to Qdrant Cloud
- **Session Storage**: Added database-based session management with file fallback
- **No Local Dependencies**: Removed reliance on local `vanna_storage` folder

### Environment Configuration

Set this environment variable for production:

```bash
SESSION_DATABASE_URL=postgresql://username:password@host:port/database_name
```

### Platform-Specific Setup:

#### **Heroku**
```bash
heroku config:set SESSION_DATABASE_URL="postgresql://username:password@host:port/database"
```

#### **Railway**
```bash
railway variables set SESSION_DATABASE_URL="postgresql://username:password@host:port/database"
```

#### **Docker**
```dockerfile
ENV SESSION_DATABASE_URL="postgresql://username:password@host:port/database"
```

#### **AWS/GCP/Azure**
Set environment variable in your platform's console.

### Database Setup

The app will automatically create a `vanna_sessions` table in your database:

```sql
CREATE TABLE vanna_sessions (
    session_id VARCHAR PRIMARY KEY,
    connection_params JSON,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### Fallback Behavior

- **With `SESSION_DATABASE_URL`**: Uses database for session storage
- **Without `SESSION_DATABASE_URL`**: Falls back to local file storage (development only)

### What You Can Delete

Now you can safely delete:
```bash
rm -rf vanna_storage/models/    # Old ChromaDB files (~11MB)
```

Keep for local development:
```bash
vanna_storage/connections/      # Local session files (fallback)
```

### Test Your Setup

1. Set `SESSION_DATABASE_URL` environment variable
2. Start your application
3. Connect to a database - session will be stored in your database
4. Ready for production deployment! 🚀

### Storage Summary

- **Vectors**: Qdrant Cloud (scalable, persistent)
- **Sessions**: Database storage (production) + file fallback (development)
- **Training Data**: Stored in Qdrant Cloud collections
- **No Local Files**: Everything is cloud-based for hosting 