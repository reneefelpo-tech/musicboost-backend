"""
Simple Email Service with Console/File Logging
For development and testing purposes
"""
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Setup logging
log_dir = Path("/app/backend/logs")
log_dir.mkdir(parents=True, exist_ok=True)

email_log_file = log_dir / "emails.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EmailService")

# File handler for email logs
file_handler = logging.FileHandler(email_log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
))
logger.addHandler(file_handler)


class EmailService:
    """
    Simple email service that logs emails to console and file
    In production, this would connect to an SMTP server
    """
    
    def __init__(self):
        self.app_name = "Music Boost"
        self.from_email = "noreply@musicboost.app"
        logger.info("EmailService initialized (Console/File logging mode)")
    
    async def send_password_reset_email(
        self, 
        to_email: str, 
        reset_code: str,
        username: Optional[str] = None
    ) -> bool:
        """
        Send password reset email with 6-digit code
        """
        subject = f"{self.app_name} - Password Reset Code"
        
        body = f"""
Hi {username or 'there'},

You recently requested to reset your password for your {self.app_name} account.

Your password reset code is: {reset_code}

This code will expire in 1 hour.

If you didn't request this, please ignore this email.

Best regards,
The {self.app_name} Team
        """
        
        return await self._send_email(to_email, subject, body)
    
    async def send_welcome_email(
        self,
        to_email: str,
        username: str,
        artist_name: str
    ) -> bool:
        """
        Send welcome email to new users
        """
        subject = f"Welcome to {self.app_name}!"
        
        body = f"""
Hi {artist_name},

Welcome to {self.app_name}! 🎵

We're excited to have you on board. Your account ({username}) has been successfully created.

Get started by:
1. Adding your music links
2. Creating viral campaigns
3. Tracking your analytics
4. Connecting with other artists

Let's boost your music career together!

Best regards,
The {self.app_name} Team
        """
        
        return await self._send_email(to_email, subject, body)
    
    async def send_campaign_notification(
        self,
        to_email: str,
        campaign_name: str,
        campaign_type: str,
        target: str
    ) -> bool:
        """
        Send notification when campaign is created
        """
        subject = f"Your {campaign_type} Campaign is Live! 🚀"
        
        body = f"""
Great news!

Your campaign "{campaign_name}" is now live and running!

Campaign Type: {campaign_type}
Target Goal: {target}

We'll keep you updated on your campaign progress.

View your campaign in the app to see real-time analytics.

Best regards,
The {self.app_name} Team
        """
        
        return await self._send_email(to_email, subject, body)
    
    async def _send_email(
        self, 
        to_email: str, 
        subject: str, 
        body: str
    ) -> bool:
        """
        Internal method to log email (simulates sending)
        """
        try:
            timestamp = datetime.utcnow().isoformat()
            
            # Log to console
            print("\n" + "="*60)
            print("📧 EMAIL SENT (Console Log)")
            print("="*60)
            print(f"From: {self.from_email}")
            print(f"To: {to_email}")
            print(f"Subject: {subject}")
            print(f"Timestamp: {timestamp}")
            print("-"*60)
            print(body)
            print("="*60 + "\n")
            
            # Log to file
            logger.info(f"EMAIL SENT | To: {to_email} | Subject: {subject}")
            logger.info(f"Body: {body.strip()}")
            logger.info("-" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log email: {str(e)}")
            return False


# Singleton instance
email_service = EmailService()
