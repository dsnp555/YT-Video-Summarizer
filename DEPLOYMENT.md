# 🚀 Deployment Guide

## Prerequisites
- GitHub account
- Streamlit Cloud account (free at https://share.streamlit.io/)
- Google Gemini API key
- Supadata API key

## Step-by-Step Deployment

### 1. Push to GitHub

```bash
# Check current status
git status

# Add all files
git add .

# Commit changes
git commit -m "Prepare for Streamlit deployment"

# Push to GitHub (if repository already exists)
git push origin main

# OR if this is a new repository:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

### 2. Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io/
   - Sign in with your GitHub account

2. **Create New App**
   - Click "New app" button
   - Select your repository
   - Choose branch: `main`
   - Main file path: `app.py`

3. **Configure Secrets**
   - Click "Advanced settings"
   - In the "Secrets" section, add:
   ```toml
   GOOGLE_API_KEY = "your_actual_google_api_key"
   SUPADATA_API_KEY = "your_actual_supadata_api_key"
   ```

4. **Deploy**
   - Click "Deploy!"
   - Wait 2-3 minutes for deployment
   - Your app will be live at: `https://YOUR_APP_NAME.streamlit.app/`

### 3. Update Your App

Whenever you make changes:
```bash
git add .
git commit -m "Your update message"
git push origin main
```

Streamlit Cloud will automatically redeploy your app!

## 🔧 Troubleshooting

### App won't start
- Check that all dependencies are in `requirements.txt`
- Verify API keys are correctly set in Streamlit secrets
- Check the logs in Streamlit Cloud dashboard

### API errors
- Ensure API keys are valid and active
- Check API quota limits
- Enable "Simple Summary" mode as fallback

### Import errors
- Make sure `requirements.txt` has all packages
- Check package versions are compatible

## 📊 Monitoring

- View logs in Streamlit Cloud dashboard
- Monitor API usage in Google AI Studio
- Check Supadata dashboard for transcript API usage

## 🎉 Success!

Your app should now be live and accessible to anyone with the URL!

Share your app: `https://YOUR_APP_NAME.streamlit.app/`
