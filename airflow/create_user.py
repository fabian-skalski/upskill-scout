#!/usr/bin/env python3
"""Create Airflow admin user programmatically for Airflow 3.x"""
import os
import sys


def create_admin_user():
    """Create admin user using FAB"""
    try:
        # First, initialize FAB security tables
        from flask import Flask
        from flask_appbuilder import AppBuilder, SQLA
        from airflow.configuration import conf
        
        username = os.environ.get("AIRFLOW_WWW_USER_USERNAME")
        password = os.environ.get("AIRFLOW_WWW_USER_PASSWORD")
        
        # Get database URL from Airflow config
        sql_alchemy_conn = os.environ.get("AIRFLOW__DATABASE__SQL_ALCHEMY_CONN")
        if not sql_alchemy_conn:
            sql_alchemy_conn = conf.get("database")
        
        # Create Flask app to initialize FAB
        app = Flask(__name__)
        app.config["SQLALCHEMY_DATABASE_URI"] = sql_alchemy_conn
        app.config["SECRET_KEY"] = "airflow-secret-key"
        
        db = SQLA(app)
        appbuilder = AppBuilder(app, db.session)
        
        # Initialize FAB tables
        with app.app_context():
            # Create all FAB tables
            db.create_all()
            
            # Check if user already exists
            user = appbuilder.sm.find_user(username=username)
            
            if user:
                print(f"User {username} already exists")
                return
            
            # Get or create Admin role
            admin_role = appbuilder.sm.find_role("Admin")
            
            if not admin_role:
                print("Admin role not found - creating it")
                admin_role = appbuilder.sm.add_role("Admin")
            
            # Create user
            user = appbuilder.sm.add_user(
                username=username,
                role=admin_role,
                password=password
            )
            
            if user:
                print(f"Successfully created admin user: {username}")
            else:
                print("Failed to create user")
                sys.exit(1)
        
    except Exception as e:
        print(f"Error creating user: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    create_admin_user()
