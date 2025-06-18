# Deploying Vanna AI to Coolify on EC2

## Prerequisites
- Coolify installed and running on your EC2 instance
- Docker installed on EC2
- Access to Coolify dashboard

## Step 1: Prepare Your Repository

1. Push your code to a Git repository (GitHub, GitLab, etc.):
```bash
git add .
git commit -m "Prepare for Coolify deployment"
git push origin main
```

## Step 2: Configure Coolify

1. **Login to Coolify Dashboard**
   - Navigate to your Coolify URL
   - Login with your credentials

2. **Add New Application**
   - Click "New Application"
   - Select "Docker Compose"
   - Choose your server (EC2 instance)

3. **Connect Repository**
   - Select your Git provider
   - Authorize Coolify to access your repository
   - Select your Vanna AI repository

4. **Configure Build**
   - Build Type: `Docker Compose`
   - Compose File: `docker-compose.yml`
   - Base Directory: `/` (root of repository)

## Step 3: Set Environment Variables in Coolify

In the Coolify application settings, add these environment variables:

```bash
DATABASE_PASSWORD=Truehorizondevmoney!
SESSION_DATABASE_URL=postgresql://postgres:Truehorizondevmoney!@rootdatabase.ctmu8c2i61mb.us-east-2.rds.amazonaws.com:5432/campaign_db
```

## Step 4: Configure Networking

1. **Port Configuration**
   - Ensure ports 8000 and 8501 are exposed
   - Coolify will automatically handle port mapping

2. **Domain Setup (Optional)**
   - Add a custom domain in Coolify
   - Configure SSL (Let's Encrypt)
   - Example domains:
     - `api.yourdomain.com` → Port 8000
     - `app.yourdomain.com` → Port 8501

## Step 5: Deploy

1. Click "Deploy" in Coolify
2. Monitor the deployment logs
3. Wait for the health check to pass

## Step 6: Verify Deployment

Once deployed, access your application:
- Backend API: `http://your-ec2-ip:8000`
- Frontend UI: `http://your-ec2-ip:8501`

Or with custom domains:
- Backend API: `https://api.yourdomain.com`
- Frontend UI: `https://app.yourdomain.com`

## Step 7: Post-Deployment Setup

1. **Connect to Database**
   - Open the frontend UI
   - Go to "🔌 Connect to Database"
   - Enter your database credentials

2. **Load Model**
   - Go to "🔥 Load Model"
   - Enter your OpenAI API key
   - The model will load from Qdrant Cloud

## Security Considerations

1. **EC2 Security Group**
   - Open ports 8000 and 8501 (or use Coolify's reverse proxy)
   - Restrict access to trusted IPs if needed

2. **Environment Variables**
   - Never commit `.env` files to Git
   - Use Coolify's secret management
   - Consider using AWS Secrets Manager for production

3. **SSL/TLS**
   - Always use HTTPS in production
   - Coolify can auto-configure Let's Encrypt

## Monitoring

1. **Coolify Dashboard**
   - Monitor application status
   - View logs
   - Check resource usage

2. **Application Health**
   - Backend health: `/` endpoint
   - Check logs for errors

## Troubleshooting

### Application Won't Start
```bash
# Check Coolify logs
# Verify environment variables are set
# Ensure database is accessible from EC2
```

### Database Connection Issues
```bash
# Test connection from EC2:
psql postgresql://postgres:password@your-rds-endpoint:5432/campaign_db
```

### Port Already in Use
```bash
# Change ports in docker-compose.yml
# Update Coolify port mappings
```

## Updating the Application

1. Push changes to Git
2. In Coolify, click "Redeploy"
3. Coolify will pull latest code and rebuild

## Backup Strategy

1. **Qdrant Cloud**: Already handles backups
2. **PostgreSQL RDS**: Enable automated backups
3. **Application Data**: Volume `vanna_storage` is persisted

## Performance Optimization

1. **EC2 Instance Size**
   - Recommended: t3.medium or larger
   - Monitor CPU and memory usage

2. **Caching**
   - Qdrant Cloud provides fast vector search
   - Consider Redis for session caching

3. **Scaling**
   - Use Coolify's scaling features
   - Consider load balancer for multiple instances 