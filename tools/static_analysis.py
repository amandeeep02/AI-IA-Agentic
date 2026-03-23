import os

def check_file(file_path):
    forbidden = ["import socket", "import requests", "os.system(", "subprocess.Popen(", "open('/dev/')"]
    warnings = ["import os", "import subprocess"]
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    issues = {"forbidden": [], "warnings": []}
    
    for issue in forbidden:
        if issue in content:
            issues["forbidden"].append(issue)
            
    for issue in warnings:
        if issue in content:
            issues["warnings"].append(issue)
            
    return issues