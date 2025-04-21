import os
import json
from flask import Flask, render_template, redirect, url_for, request, jsonify, session
from access_log import clear_logs, log_access

def setup_clear_logs_routes(app):
    """
    Set up routes for clearing logs.
    This function should be called from app.py.
    """
    
    @app.route('/logs/clear', methods=['GET', 'POST'])
    def clear_logs_route():
        """Clear all logs."""
        if not session.get('admin_logged_in'):
            if request.method == 'GET':
                return redirect(url_for('admin_login'))
            else:
                return jsonify({'success': False, 'message': 'Unauthorized'}), 401
        
        try:
            # Clear all logs
            print("Attempting to clear logs...")
            clear_logs()
            print("Logs cleared successfully")
            
            # Log this action
            print("Logging the clear action...")
            log_access("Admin", True, "Cleared Access Logs")
            print("Clear action logged successfully")
            
            # Handle different response types based on request method
            if request.method == 'GET':
                # Redirect back to access logs page for direct links
                return redirect(url_for('access_logs'))
            else:
                # Return JSON for API requests
                return jsonify({'success': True, 'message': 'All logs have been cleared successfully'})
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error clearing logs: {str(e)}")
            print(f"Error details: {error_details}")
            
            if request.method == 'GET':
                # Redirect with error for direct links
                return redirect(url_for('access_logs'))
            else:
                # Return JSON error for API requests
                return jsonify({'success': False, 'message': f'Error clearing logs: {str(e)}'}), 500
