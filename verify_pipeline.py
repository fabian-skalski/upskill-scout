#!/usr/bin/env python3
"""Simple pipeline verification with parallel job submission."""
import requests
import time
import random
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import numpy as np

BACKEND_URL = "http://localhost:8000"
USER_ID = "test_user"

JOB_DESCR = [
    "Senior Python Developer. We are looking for a Senior Python Developer to build scalable backend services using FastAPI and Django. You will work with PostgreSQL, Redis, and Docker. Experience with cloud platforms like AWS is a plus. Responsibilities include designing APIs, optimizing database queries, and mentoring junior developers.",
    "Data Scientist. Join our data team to build predictive models and analyze large datasets. Proficiency in Python, SQL, and libraries like pandas, scikit-learn, and TensorFlow is required. You will collaborate with product teams to drive data-driven decision making.",
    "Frontend Engineer. We need a skilled Frontend Engineer to create responsive and interactive user interfaces using React and TypeScript. You should have a strong understanding of HTML, CSS, and modern JavaScript. Experience with state management libraries like Redux or Zustand is preferred.",
    "DevOps Engineer. We are seeking a DevOps Engineer to manage our CI/CD pipelines and cloud infrastructure. You will work with Kubernetes, Terraform, and Jenkins. Your goal is to ensure high availability and reliability of our services.",
    "Full Stack Engineer. Looking for a Full Stack Engineer comfortable with both frontend and backend technologies. You will work with a stack comprising React, Node.js, and MongoDB. Experience with GraphQL and REST APIs is essential.",
    "Machine Learning Engineer. We are hiring a Machine Learning Engineer to deploy and maintain ML models in production. You should be proficient in Python and have experience with MLOps tools like MLflow and Kubeflow. Knowledge of deep learning frameworks is a must.",
    "Cloud Architect. We are looking for a Cloud Architect to design secure and scalable cloud solutions on Azure. You will define cloud strategy, oversee migration projects, and ensure compliance with security standards. Certifications in Azure are highly desirable.",
    "Cybersecurity Analyst. Join our security team to monitor and protect our systems from cyber threats. You will conduct vulnerability assessments, analyze security logs, and respond to incidents. Knowledge of SIEM tools and network security protocols is required.",
    "Technical Product Manager. We need a Technical Product Manager to lead our engineering teams. You will define product roadmaps, gather requirements, and prioritize features. A background in software development and experience with Agile methodologies is preferred.",
    "QA Automation Engineer. We are seeking a QA Automation Engineer to build and maintain automated test suites. You will use tools like Selenium, Pytest, and Appium. Your role is to ensure the quality and stability of our software releases."
]

SHOULD_SUBMIT_NEW_JOBS = True
SHOULD_COMPUTE_USERS_OVERVIEW = True


def submit_job(_):
    try:
        requests.post(f"{BACKEND_URL}/text", json={
            "description": random.choice(JOB_DESCR),
            "sourceUrl": f"https://ex.com/job/{random.randint(1000,9999)}",
            "timestamp": f"2025-11-{random.randint(20,29):02d}T10:00:00Z",
            "user_id": USER_ID
        }, timeout=10).raise_for_status()
        return "✓"
    except:
        return "✗"

def main():
    import sys
    n = 10
    
    print("="*70)
    print(f"VERIFICATION | User: {USER_ID} | Jobs: {n}")
    print("="*70)

    print(f"\n📤 Submitting {n} jobs...")
    if SHOULD_SUBMIT_NEW_JOBS:
        with ThreadPoolExecutor(max_workers=10) as ex:
            results = list(ex.map(submit_job, range(n)))
        print(f"✓ {results.count('✓')}/{n} submitted")
    else:
        print("Skipping job submission as per configuration.")
    
    if SHOULD_COMPUTE_USERS_OVERVIEW:
        print(f"\n⏳ Waiting for jobs to process (polling every 1s)...")
        elapsed = 0
        while True:
            time.sleep(1)
            elapsed += 1
            # Check if jobs are ready by trying to trigger overview
            try:
                r = requests.post(f"{BACKEND_URL}/overview", json={"user_id": USER_ID})
                if r.status_code == 409:
                    print(f"✓ Processing started after {elapsed}s")
                    break
                elif r.status_code in [200, 201, 202]:
                    print(f"✓ Triggered clustering after {elapsed}s")
                    break
            except:
                pass
            if elapsed % 10 == 0:
                print(f"  ...{elapsed}s elapsed")

    print("\n⏳ Waiting for results (polling every 1s, retrying on errors)...")
    poll_count = 0
    while True:
        poll_count += 1
        time.sleep(1)
        
        try:
            r = requests.get(f"{BACKEND_URL}/overview", params={"user_id": USER_ID})
            
            # Retry infinitely on 4XX/5XX errors
            if r.status_code >= 400:
                if poll_count % 10 == 0:
                    print(f"  ...{poll_count}s: Status {r.status_code}, retrying...")
                continue
            
            # Check for successful response with data
            if r.status_code == 200 and r.json():
                data = r.json()
                if data and "umap_points" in data[0]:
                    print(f"✓ {len(data)} clusters found after {poll_count}s")
                    
                    # Visualize with heatmap colors based on relevancy
                    data = sorted(data, key=lambda x: x["relevancy_score"], reverse=True)
                    _, ax = plt.subplots(figsize=(14, 10))
                    
                    # Create heatmap colormap (single direction: low to high relevancy)
                    from matplotlib.colors import LinearSegmentedColormap
                    cmap = plt.cm.Reds  # White (low) -> Red (high) - single direction
                    
                    # Get relevancy scores and normalize them
                    relevancy_scores = [c["relevancy_score"] for c in data]
                    min_rel = min(relevancy_scores)
                    max_rel = max(relevancy_scores)

                    # Add colorbar legend on the right side
                    from matplotlib.cm import ScalarMappable
                    from matplotlib.colors import Normalize
                    
                    norm = Normalize(vmin=min_rel, vmax=max_rel)
                    sm = ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    
                    # Create colorbar
                    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.02, shrink=0.8, aspect=20)
                    cbar.set_label('Relevancy Score (%)', labelpad=20, fontsize=11, weight='bold')
                    cbar.ax.tick_params(labelsize=9)
                    
                    # Plot each cluster with color based on relevancy
                    scatter_plots = []
                    for idx, c in enumerate(data):
                        pts = c["umap_points"]
                        if pts:
                            # Extract coordinates (now using 'coordinates' field)
                            # Support both 2D and 3D+ (but only plot first 2 dimensions)
                            x = [p["coordinates"][0] for p in pts]
                            y = [p["coordinates"][1] for p in pts]
                            
                            # Calculate cluster center for label placement
                            center_x = np.mean(x)
                            center_y = np.mean(y)
                            
                            # Normalize relevancy score to 0-1 for colormap
                            rel_score = c["relevancy_score"]
                            if max_rel > min_rel:
                                norm_score = (rel_score - min_rel) / (max_rel - min_rel)
                            else:
                                norm_score = 0.5
                            
                            color = cmap(norm_score)
                            
                            # Plot points without individual labels
                            scatter = ax.scatter(x, y, c=[color], s=100, alpha=0.7, 
                                               edgecolors='white', linewidths=0.5)
                            scatter_plots.append(scatter)
                            
                            # Add text label above the circle with highest z-index
                            label_text = f"{c.get('description', 'Unknown')}: {c['relevancy_score']:.1f}%"
                            ax.text(center_x, center_y + 0.3, label_text, 
                                   fontsize=9, weight='bold', 
                                   ha='center', va='bottom',
                                   bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.35, edgecolor='none'),
                                   zorder=1000)

                    # Maximize zoom - set tighter axis limits
                    all_x = [p["coordinates"][0] for c in data for p in c["umap_points"]]
                    all_y = [p["coordinates"][1] for c in data for p in c["umap_points"]]
                    
                    x_range = max(all_x) - min(all_x)
                    y_range = max(all_y) - min(all_y)
                    
                    # Add small padding (5% of range)
                    padding = 0.05
                    ax.set_xlim(min(all_x) - x_range * padding, max(all_x) + x_range * padding)
                    ax.set_ylim(min(all_y) - y_range * padding, max(all_y) + y_range * padding)
                                       
                    # Labels and title
                    ax.set_xlabel('Latent Dimension 1', fontsize=11, weight='bold')
                    ax.set_ylabel('Latent Dimension 2', fontsize=11, weight='bold')
                    ax.set_title(f'Desirable Skill Clusters for {USER_ID}\n(Color: White=Low Relevancy → Red=High Relevancy)', 
                               fontsize=13, weight='bold', pad=20)
                    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                    
                    plt.tight_layout()
                    plt.savefig(f"clusters_{USER_ID}.png", dpi=300, bbox_inches='tight')
                    print(f"✓ Saved clusters_{USER_ID}.png")
                    plt.show()
                    print("\n✓ COMPLETE!")
                    return
            
            if poll_count % 10 == 0:
                print(f"  ...{poll_count}s: waiting for clustering to complete")
                
        except Exception as e:
            # Retry on any exception (connection errors, etc.)
            if poll_count % 10 == 0:
                print(f"  ...{poll_count}s: Error ({type(e).__name__}), retrying...")

if __name__ == "__main__":
    main()
