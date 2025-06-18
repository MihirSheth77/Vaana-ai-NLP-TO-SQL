# Deployment Checklist for Coolify

## Pre-Deployment
- [ ] All code changes are committed and pushed to Git
- [ ] Environment variables are documented
- [ ] Docker builds successfully locally
- [ ] Application runs correctly locally

## Coolify Setup
- [ ] Coolify is accessible on EC2
- [ ] Git repository is connected to Coolify
- [ ] Environment variables are set in Coolify:
  - [ ] `DATABASE_PASSWORD`
  - [ ] `SESSION_DATABASE_URL` (optional)
- [ ] Docker Compose configuration is selected
- [ ] Ports 8000 and 8501 are configured

## EC2 Configuration
- [ ] Security Group allows inbound traffic on:
  - [ ] Port 8000 (Backend API)
  - [ ] Port 8501 (Frontend UI)
  - [ ] Port 22 (SSH - restricted to your IP)
- [ ] EC2 instance has sufficient resources (t3.medium or larger)
- [ ] Docker is installed on EC2
- [ ] Coolify is running

## Database Connectivity
- [ ] RDS PostgreSQL is accessible from EC2
- [ ] Security group allows EC2 to connect to RDS
- [ ] Database credentials are correct

## Deployment
- [ ] Deploy button clicked in Coolify
- [ ] Build logs show no errors
- [ ] Health checks are passing
- [ ] Application is accessible via EC2 IP

## Post-Deployment Verification
- [ ] Backend API responds at `http://ec2-ip:8000/`
- [ ] Frontend UI loads at `http://ec2-ip:8501/`
- [ ] Can connect to database from UI
- [ ] Can load model with OpenAI API key
- [ ] Can perform queries successfully

## Security
- [ ] No sensitive data in Git repository
- [ ] Environment variables are secure in Coolify
- [ ] HTTPS is configured (if using custom domain)
- [ ] Firewall rules are restrictive

## Monitoring
- [ ] Coolify monitoring is active
- [ ] Application logs are accessible
- [ ] Resource usage is being tracked

## Documentation
- [ ] Team knows how to access the application
- [ ] Deployment process is documented
- [ ] Troubleshooting guide is available 