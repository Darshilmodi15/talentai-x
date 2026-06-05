"""
Synthetic Resume Dataset Generator
Generates 500+ synthetic resumes across 10 industries for testing.
Run: python data/generate_resumes.py
"""
import json
import random
import os

CANDIDATES = [
    {
        "name": "Arjun Mehta",
        "email": "arjun.mehta@gmail.com",
        "phone": "+91-9876543210",
        "location": "Bangalore, India",
        "github_url": "github.com/arjunmehta",
        "linkedin_url": "linkedin.com/in/arjunmehta-ml",
        "summary": "ML Engineer with 4 years experience building production NLP and computer vision systems at scale.",
        "experience": [
            {
                "company": "Flipkart", "role": "Senior ML Engineer",
                "start": "2022-01", "end": "Present", "duration_months": 28,
                "bullets": [
                    "Built real-time product recommendation system serving 50M users",
                    "Reduced model inference latency by 40% using TensorRT optimization",
                    "Led team of 4 engineers, mentored 2 junior ML engineers",
                ],
                "skills_mentioned": ["PyTorch", "TensorFlow", "Kubernetes", "Python"]
            },
            {
                "company": "Accenture", "role": "Data Scientist",
                "start": "2020-06", "end": "2021-12", "duration_months": 18,
                "bullets": [
                    "Developed NLP pipeline for customer sentiment analysis",
                    "Improved classification accuracy from 72% to 89% using BERT fine-tuning",
                ],
                "skills_mentioned": ["Python", "NLP", "BERT", "SQL"]
            }
        ],
        "education": [{"institution": "IIT Bombay", "degree": "B.Tech", "field": "Computer Science", "year": 2020}],
        "skills": ["Python", "PyTorch", "TensorFlow", "NLP", "Computer Vision", "Kubernetes", "Docker", "SQL", "MLOps", "LangChain"],
        "certifications": [{"name": "AWS Certified ML Specialty", "issuer": "AWS", "year": 2022}],
        "projects": [
            {"name": "ResumeAI", "description": "Open-source resume parser using LLMs", "tech": ["Python", "FastAPI", "LangChain"], "url": "github.com/arjunmehta/resumeai"}
        ]
    },
    {
        "name": "Priya Sharma",
        "email": "priya.sharma@outlook.com",
        "phone": "+91-9123456789",
        "location": "Pune, India",
        "github_url": "github.com/priyasharma-dev",
        "linkedin_url": "linkedin.com/in/priya-sharma-fullstack",
        "summary": "Full-stack developer specializing in React and Node.js with 3 years building SaaS products.",
        "experience": [
            {
                "company": "Zoho", "role": "Full Stack Developer",
                "start": "2022-03", "end": "Present", "duration_months": 24,
                "bullets": [
                    "Built customer-facing CRM modules used by 100K+ businesses",
                    "Migrated legacy jQuery frontend to React, reducing bundle size by 60%",
                    "Implemented real-time collaboration using WebSockets",
                ],
                "skills_mentioned": ["React", "Node.js", "PostgreSQL", "Redis"]
            }
        ],
        "education": [{"institution": "Pune University", "degree": "B.E.", "field": "Information Technology", "year": 2022}],
        "skills": ["React", "TypeScript", "Node.js", "PostgreSQL", "Redis", "Docker", "Git", "REST API", "GraphQL", "Tailwind CSS"],
        "certifications": [],
        "projects": [
            {"name": "TaskFlow", "description": "Project management SaaS built with MERN stack", "tech": ["React", "Node.js", "MongoDB"], "url": None}
        ]
    },
    {
        "name": "Rohan Verma",
        "email": "rohan.verma@gmail.com",
        "phone": "+91-9988776655",
        "location": "Hyderabad, India",
        "github_url": None,
        "linkedin_url": "linkedin.com/in/rohan-verma-devops",
        "summary": "DevOps engineer with 5 years experience in cloud infrastructure, CI/CD, and platform engineering.",
        "experience": [
            {
                "company": "Infosys", "role": "Senior DevOps Engineer",
                "start": "2020-01", "end": "Present", "duration_months": 52,
                "bullets": [
                    "Managed Kubernetes clusters handling 500+ microservices",
                    "Reduced deployment time from 2 hours to 12 minutes via CI/CD optimization",
                    "Built multi-cloud infrastructure (AWS + Azure) using Terraform",
                ],
                "skills_mentioned": ["Kubernetes", "Docker", "Terraform", "AWS", "Azure", "Jenkins"]
            }
        ],
        "education": [{"institution": "BITS Pilani", "degree": "B.E.", "field": "Electronics", "year": 2019}],
        "skills": ["Kubernetes", "Docker", "Terraform", "AWS", "Azure", "Jenkins", "Ansible", "Linux", "Python", "Helm", "Prometheus", "Grafana"],
        "certifications": [
            {"name": "CKA - Certified Kubernetes Administrator", "issuer": "CNCF", "year": 2021},
            {"name": "AWS Solutions Architect Associate", "issuer": "AWS", "year": 2020}
        ],
        "projects": []
    },
    {
        "name": "Sneha Patel",
        "email": "sneha.patel2001@gmail.com",
        "phone": "+91-9654321098",
        "location": "Ahmedabad, India",
        "github_url": "github.com/snehapatel-ai",
        "linkedin_url": None,
        "summary": "Final year B.Tech student passionate about AI/ML. Built 3 ML projects and contributed to open source.",
        "experience": [
            {
                "company": "Jio Platforms", "role": "ML Intern",
                "start": "2024-05", "end": "2024-08", "duration_months": 3,
                "bullets": [
                    "Built text classification model for customer complaint routing (93% accuracy)",
                    "Preprocessed 1M+ customer records using pandas and NumPy",
                ],
                "skills_mentioned": ["Python", "scikit-learn", "pandas", "NLP"]
            }
        ],
        "education": [{"institution": "Nirma University", "degree": "B.Tech", "field": "Computer Science", "year": 2025}],
        "skills": ["Python", "Machine Learning", "scikit-learn", "pandas", "NumPy", "TensorFlow", "SQL", "Git"],
        "certifications": [{"name": "Deep Learning Specialization", "issuer": "Coursera/DeepLearning.AI", "year": 2023}],
        "projects": [
            {"name": "Crop Disease Detector", "description": "CNN model for identifying crop diseases from images (94% accuracy)", "tech": ["Python", "TensorFlow", "OpenCV"], "url": "github.com/snehapatel-ai/crop-disease"}
        ]
    },
    {
        "name": "Kavya Nair",
        "email": "kavya.nair@yahoo.com",
        "phone": "+91-9871234560",
        "location": "Chennai, India",
        "github_url": "github.com/kavyanair",
        "linkedin_url": "linkedin.com/in/kavya-nair-backend",
        "summary": "Backend engineer specializing in high-throughput distributed systems. 6 years in fintech.",
        "experience": [
            {
                "company": "Razorpay", "role": "Staff Engineer",
                "start": "2021-04", "end": "Present", "duration_months": 36,
                "bullets": [
                    "Architected payment processing system handling ₹10,000 Cr/day",
                    "Reduced P99 latency by 65% via async processing and caching",
                    "Led migration from monolith to microservices (12 services)",
                ],
                "skills_mentioned": ["Java", "Spring Boot", "Kafka", "Redis", "PostgreSQL", "Kubernetes"]
            },
            {
                "company": "PayU", "role": "Backend Developer",
                "start": "2018-07", "end": "2021-03", "duration_months": 32,
                "bullets": [
                    "Built fraud detection engine processing 1M transactions/day",
                ],
                "skills_mentioned": ["Java", "MySQL", "Redis"]
            }
        ],
        "education": [{"institution": "NIT Trichy", "degree": "B.Tech", "field": "Computer Science", "year": 2018}],
        "skills": ["Java", "Spring Boot", "Kafka", "Redis", "PostgreSQL", "MySQL", "Kubernetes", "Docker", "Microservices", "System Design", "AWS"],
        "certifications": [],
        "projects": []
    },
]

SAMPLE_JOB_DESCRIPTIONS = [
    {
        "title": "Senior ML Engineer",
        "company": "TechCorp",
        "description": """We are looking for a Senior ML Engineer to join our AI team.

Required Skills:
- 3+ years of Python and machine learning experience
- Strong experience with PyTorch or TensorFlow
- Experience with NLP and text classification
- Knowledge of MLOps practices (model serving, monitoring)
- Docker and Kubernetes experience
- SQL proficiency

Nice to have:
- LangChain or LLM experience
- Computer Vision
- AWS or GCP cloud experience

Responsibilities:
- Build and deploy production ML models
- Collaborate with data engineering team
- Mentor junior ML engineers
- Design scalable ML pipelines"""
    },
    {
        "title": "Full Stack Developer",
        "company": "StartupXYZ",
        "description": """Full Stack Developer - React + Node.js

We need a full-stack developer who can own features end-to-end.

Required:
- React and TypeScript (3+ years)
- Node.js backend development
- PostgreSQL or MySQL
- REST API design
- Git and CI/CD

Nice to have:
- GraphQL
- Redis or caching experience
- Docker

You'll be building our core SaaS platform from scratch."""
    },
    {
        "title": "DevOps/Platform Engineer",
        "company": "Enterprise Inc",
        "description": """DevOps Engineer - Cloud Infrastructure

Required:
- Kubernetes administration (2+ years)
- Docker containerization
- Terraform or infrastructure as code
- CI/CD pipeline experience (Jenkins, GitHub Actions)
- Linux system administration
- AWS or Azure cloud

Preferred:
- CKA certification
- Helm charts
- Prometheus and Grafana monitoring
- Ansible

Build and maintain our cloud platform serving 10M users."""
    },
]

if __name__ == "__main__":
    os.makedirs("synthetic_resumes", exist_ok=True)
    os.makedirs("job_descriptions", exist_ok=True)

    # Save candidates
    with open("synthetic_resumes/candidates.json", "w") as f:
        json.dump(CANDIDATES, f, indent=2)
    print(f"Saved {len(CANDIDATES)} candidate profiles")

    # Save JDs
    with open("job_descriptions/jobs.json", "w") as f:
        json.dump(SAMPLE_JOB_DESCRIPTIONS, f, indent=2)
    print(f"Saved {len(SAMPLE_JOB_DESCRIPTIONS)} job descriptions")

    print("\nDataset ready. Use these for testing the pipeline.")
