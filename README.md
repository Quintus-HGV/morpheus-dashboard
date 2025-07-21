# Morpheus User Insights Dashboard

*Visualizing user journeys and cloud platform usage patterns*

## Introduction

The Morpheus User Insights Dashboard is an advanced analytics platform designed for HPE's Morpheus cloud management solution. This tool transforms raw user activity data into actionable insights, helping platform administrators understand user behaviors, optimize resource allocation, and improve customer experiences.

Cloud platforms traditionally lack visibility into end-to-end user journeys, focusing instead on isolated actions like VM creation. This limitation hinders meaningful UX improvements and resource optimization. Our dashboard solves this by:

- Tracking complete user journeys across provisioning, execution, and resource usage
- Identifying recurring usage patterns and anomalies
- Providing AI-powered insights into temporal behavior
- Visualizing resource utilization and engagement metrics

## Key Features

### 📊 Comprehensive Activity Analysis
- Tenant-wise activity timelines with success/failure indicators
- Weekly and daily activity breakdowns
- Hourly peak usage analysis
- Month-over-MoM change metrics

### 🤖 AI-Powered Insights
- Tenant behavior analysis using DeepSeek AI
- Temporal pattern detection (anomalies, inefficiencies, trends)
- Predictive resource allocation recommendations

### 👥 User Engagement Metrics
- Top engaged user identification
- At-risk user detection
- Retention analysis
- BCG matrix for VM provisioning vs CPU usage

### ⚙️ Resource Management
- Instance type distribution analysis
- Tenant-level Gantt charts for run activities
- CPU utilization heatmaps
- Provisioning/deletion tracking

## Technologies Used

**Backend & Data**
- Neo4j Graph Database
- Python 3.9+
- Pandas for data processing
- Plotly for visualizations
- DeepSeek AI for insights generation

**Frontend**
- Streamlit for dashboard UI
- Plotly for interactive charts
- Custom CSS for dark theme styling

## Getting Started

### Prerequisites
- Python 3.9+
- Neo4j database (v4.4+)
- Docker (optional for Neo4j)
- API key for [OpenRouter](https://openrouter.ai/)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/syed0369/morpheus-user-insights.git
cd morpheus-user-insights
```

2. **Create and activate virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
Create a `.env` file with your credentials:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
OPENROUTER_API_KEY=your_api_key
```

### Running the Application
```bash
streamlit run main.py
```

The dashboard will launch at `http://localhost:8501`

## Project Structure

```
morpheus-user-insights/
├── main.py                 # Application entry point
├── setup.py                # UI configuration and setup
├── views.py                # Dashboard views and visualizations
├── load_data.py            # Neo4j data loading functions
├── .env.example            # Environment variable template
├── requirements.txt        # Python dependencies
├── README.md               # This documentation
└── .gitignore
```

## Key Components

### 1. Data Loading (`load_data.py`)
- Connects to Neo4j graph database
- Retrieves user actions, executions, and runs
- Prepares data for LLM analysis
- Caches queries for performance

### 2. Dashboard Setup (`setup.py`)
- Configures Streamlit page settings
- Creates sidebar filters (tenants, date ranges)
- Implements custom tooltip system
- Handles data filtering

### 3. Visualizations (`views.py`)
- **Activity Timeline**: Tenant-wise activity scatter plot
- **Success/Failure Analysis**: Action outcome tracking
- **BCG Matrix**: VM provisioning vs CPU usage
- **Retention Analysis**: User engagement over time
- **AI Insights**: LLM-generated behavior analysis
- **Gantt Charts**: Resource utilization timelines

## Usage Guide

### Basic Navigation
1. **Sidebar Filters**:
   - Select tenants to analyze
   - Adjust date range for historical analysis
   - Choose specific weeks for daily breakdowns

2. **Dashboard Sections**:
   - Top: AI-generated insights and key metrics
   - Middle: Activity timelines and trends
   - Bottom: Resource utilization and user analytics

3. **Interactive Features**:
   - Hover over charts for detailed information
   - Click legend items to toggle visibility
   - Use expanders to drill down into user-specific data

### Generating Insights
1. Tenant behavior insights are automatically generated when tenants are selected
2. For custom queries, use the chatbot at the bottom of the dashboard
3. Adjust time ranges to compare different periods

## Acknowledgments
- HPE Morpheus team for domain expertise
- Neo4j for graph database support
- Streamlit for rapid dashboard development
- OpenRouter for LLM API access

## Contributors
- Harsh Gupta (harshgupta.is22@rvce.edu.in)
- Manish Raj (manishsraj.is22@rvce.edu.in)
- Sanjana Bhagwath (sanjanab.is22@rvce.edu.in)
- Syed Umair (syedumair.is22@rvce.edu.in)
- Yashvanth B L (yashvanthbl.is22@rvce.edu.in)
---

**Empowering cloud platform administrators with actionable insights since 2025**  
*Part of HPE's mission to improve customer experience through data-driven decisions*
