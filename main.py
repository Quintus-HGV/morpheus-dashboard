import views
import setup
from load_data import load_combined_data

if __name__ == "__main__":
    setup.setup()
    df = load_combined_data()
    selected_tenants, date_range = setup.setup_sidebar(df)
    
    # AI Insights Section
    views.insights(selected_tenants)
    
    # Filter data based on selections
    filtered_df = setup.filter_data(df, selected_tenants, date_range)
    
    # New Activity Metrics Comparison
    views.tenant_comparison(selected_tenants, filtered_df)
    
    # Existing visualizations
    views.display_activity_chart(filtered_df)
    views.display_action_success_chart(filtered_df)
    pivot = views.display_weekly_activity(filtered_df)
    combined_daywise, selected_weeks, available_weeks, select_all = views.display_daily_activity(
        pivot, filtered_df, selected_tenants
    )
    
    # User Insights
    views.user_engagement_insights(filtered_df)
    #views.user_activity_trends_stacked(filtered_df)
    views.user_activity_trends_simple(filtered_df)
    views.display_top_active_users(combined_daywise)
    
    # Resource Management Views
    views.display_bcg_matrix()
    views.instance_type_distribution(selected_tenants, date_range, selected_weeks, available_weeks, select_all)
    views.display_tenant_gantt_chart(selected_tenants, date_range, selected_weeks, available_weeks, select_all)
    
    # Chatbot
    views.chatbot_ui()