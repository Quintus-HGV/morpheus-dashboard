import streamlit as st
import pandas as pd
from llm import rag_engine
import plotly.express as px
import plotly.graph_objects as go
import datetime
from itertools import product
import json
import numpy as np
from setup import graph_heading_with_info
from load_data import fetch_instance_counts, fetch_run_data, fetch_temporal_activity_data, get_temporal_insights_from_ai, prepare_llm_friendly_json

# --- New: Activity Metrics Comparison ---
def tenant_comparison(selected_tenants, filtered_df):
    st.subheader("📊 Tenant Comparison Dashboard")
    
    # Activity comparison metrics
    st.markdown("### Activity Metrics Comparison")
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    # Calculate overall metrics
    total_actions = filtered_df.shape[0]
    active_tenants = len(selected_tenants)
    
    # Calculate MoM change
    today = datetime.date.today()
    current_month_start = today.replace(day=1)
    prev_month_end = current_month_start - datetime.timedelta(days=1)
    prev_month_start = prev_month_end.replace(day=1)
    
    current_month_data = filtered_df[
        (filtered_df["date"] >= current_month_start) & 
        (filtered_df["date"] <= today)
    ]
    
    prev_month_data = filtered_df[
        (filtered_df["date"] >= prev_month_start) & 
        (filtered_df["date"] <= prev_month_end)
    ]
    
    current_activity = current_month_data.shape[0]
    prev_activity = prev_month_data.shape[0]
    
    mom_change = ((current_activity - prev_activity) / prev_activity * 100) if prev_activity > 0 else float('inf')
    
    # Display metrics
    with col1:
        st.metric("Total Actions", total_actions)
    with col2:
        st.metric("Active Tenants", active_tenants)
    with col3:
        display_value = f"{mom_change:.1f}%" if not np.isinf(mom_change) else "N/A (New)"
        delta = mom_change if not np.isinf(mom_change) else None
        st.metric("MoM Change", display_value, delta=f"{delta:.1f}%" if delta else None)
    
    # Create tenant comparison table
    activity_comparison = filtered_df.groupby("tenant").agg(
        total_actions=("type", "count"),
        unique_users=("username", "nunique"),
        last_activity=("ts", "max")
    ).reset_index()
    
    # Calculate MoM change per tenant
    current_counts = current_month_data.groupby("tenant").size()
    prev_counts = prev_month_data.groupby("tenant").size()
    
    activity_comparison["mom_change"] = activity_comparison["tenant"].apply(
        lambda x: ((current_counts.get(x, 0) - prev_counts.get(x, 0)) / 
                 prev_counts.get(x, 1) * 100 if prev_counts.get(x, 0) > 0 
                 else float('inf'))
    )
    
    # Format and display table
    formatted_df = activity_comparison.copy()
    formatted_df["last_activity"] = formatted_df["last_activity"].dt.strftime("%Y-%m-%d")
    formatted_df["mom_change"] = formatted_df["mom_change"].apply(
        lambda x: f"{x:.1f}%" if not np.isinf(x) else "N/A (New)"
    )
    
    st.dataframe(
        formatted_df.rename(columns={
            "tenant": "Tenant",
            "total_actions": "Total Actions",
            "unique_users": "Unique Users",
            "last_activity": "Last Activity",
            "mom_change": "MoM Change"
        }).style.applymap(
            lambda x: 'color: red' if isinstance(x, str) and '-' in x else 'color: green',
            subset=["MoM Change"]
        ),
        use_container_width=True
    )
    
    # --- Activity Trend Comparison ---
    st.markdown("### Activity Trend Comparison")
    daily_activity = filtered_df.groupby(["date", "tenant"]).size().unstack().fillna(0)
    
    fig = px.line(
        daily_activity,
        x=daily_activity.index,
        y=daily_activity.columns,
        title="Daily Activity Trend Comparison",
        labels={"value": "Activity Count", "date": "Date", "variable": "Tenant"},
        height=500
    )
    fig.update_layout(
        hovermode="x unified",
        legend_title_text="Tenant",
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font_color="white"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Peak Usage Analysis ---
    st.markdown("### Peak Usage Analysis")
    hourly_activity = filtered_df.copy()
    hourly_activity["hour"] = hourly_activity["ts"].dt.hour
    hourly_counts = hourly_activity.groupby(["tenant", "hour"]).size().reset_index(name="count")
    
    fig = px.bar(
        hourly_counts,
        x="hour",
        y="count",
        color="tenant",
        barmode="group",
        title="Hourly Activity Distribution",
        labels={"count": "Activity Count", "hour": "Hour of Day"},
        height=500
    )
    fig.update_layout(
        xaxis=dict(tickmode="linear", dtick=1),
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font_color="white"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- New: User Engagement Insights ---
def user_engagement_insights(filtered_df):
    graph_heading_with_info(
        "User Engagement Insights",
        "Identifies top engaged users and at-risk users based on activity patterns"
    )
    
    # User engagement metrics
    st.markdown("### User Engagement Metrics")
    user_stats = filtered_df.groupby(["tenant", "username"]).agg(
        total_actions=("type", "count"),
        last_activity=("ts", "max"),
        active_days=("date", "nunique")
    ).reset_index()
    
    # Calculate engagement scores
    max_date = filtered_df["date"].max()
    user_stats["last_activity"] = pd.to_datetime(user_stats["last_activity"])
    user_stats["days_since_active"] = (max_date - user_stats["last_activity"].dt.date).apply(lambda x: x.days)
    user_stats["engagement_score"] = user_stats["total_actions"] / user_stats["active_days"]
    
    # Display top/bottom users
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top Engaged Users**")
        top_users = user_stats.nlargest(5, "engagement_score")
        st.dataframe(top_users[["tenant", "username", "engagement_score"]].rename(columns={
            "tenant": "Tenant", "username": "User", "engagement_score": "Engagement Score"
        }), hide_index=True)
    
    with col2:
        st.markdown("**At-Risk Users**")
        at_risk = user_stats[user_stats["days_since_active"] > 14].nlargest(5, "days_since_active")
        st.dataframe(at_risk[["tenant", "username", "days_since_active"]].rename(columns={
            "tenant": "Tenant", "username": "User", "days_since_active": "Days Inactive"
        }), hide_index=True)
    
    # --- User Retention Analysis ---
    graph_heading_with_info(
        "User Retention Analysis",
        "Shows user retention rates over time with simplified visualization"
    )
    
    retention_df = filtered_df.copy()
    retention_df["date"] = pd.to_datetime(retention_df["date"], errors="coerce")
    retention_df["cohort_month"] = (
    retention_df.groupby("username")["date"]
    .transform("min")
    .dt.to_period("M")
    )
    retention_df["activity_month"] = retention_df["date"].dt.to_period("M")
    
    # Calculate retention rates
    cohort_data = retention_df.groupby(["cohort_month", "activity_month"]).agg(
        users=("username", "nunique")
    ).reset_index()
    
    cohort_size = cohort_data.groupby("cohort_month")["users"].first().reset_index()
    cohort_size.columns = ["cohort_month", "cohort_size"]
    
    retention_rates = pd.merge(cohort_data, cohort_size, on="cohort_month")
    retention_rates["retention_rate"] = (retention_rates["users"] / retention_rates["cohort_size"]) * 100
    
    # Calculate months since joining
    retention_rates["months_since_joining"] = (
        retention_rates["activity_month"] - retention_rates["cohort_month"]
    ).apply(lambda x: x.n)
    
    # Simplify to line chart
    fig = px.line(
        retention_rates,
        x="months_since_joining",
        y="retention_rate",
        color="cohort_month",
        title="User Retention Over Time",
        labels={
            "retention_rate": "Retention Rate (%)",
            "months_since_joining": "Months After Joining",
            "cohort_month": "Cohort Month"
        },
        height=500
    )
    fig.update_layout(
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font_color="white",
        legend_title_text="Cohort Month"
    )
    st.plotly_chart(fig, use_container_width=True)


# --- NEW: Simple Weekly User Activity ---
def user_activity_trends_simple(filtered_df):
    """
    Weekly user activity (absolute counts) shown as small-multiples bar charts.
    Much cleaner than the percentage-stacked monster.
    """
    graph_heading_with_info(
        "Weekly User Activity Count",
        "Each mini-chart shows how many actions each user performed per week. "
        "Easier to read and compare across tenants."
    )

    # 1. keep only last 8 weeks to avoid clutter
    cutoff = datetime.date.today() - datetime.timedelta(weeks=8)
    df = filtered_df[filtered_df["date"] >= cutoff].copy()

    # 2. week start (Monday)
    df["week_start"] = df["ts"] - pd.to_timedelta(df["ts"].dt.weekday, unit="D")
    df["week_start"] = df["week_start"].dt.date

    # 3. counts per user per week
    counts = (
        df.groupby(["week_start", "tenant", "username"])
        .size()
        .reset_index(name="activity_count")
    )

    # 4. small-multiples bar chart
    fig = px.bar(
        counts,
        x="week_start",
        y="activity_count",
        color="username",
        facet_col="tenant",
        facet_col_wrap=2,
        height=600,
        labels={
            "activity_count": "Actions",
            "week_start": "Week Starting",
            "username": "User",
        },
        title="Weekly Actions per User"
    )

    fig.update_xaxes(tickformat="%b %d", tickangle=-45)
    fig.update_layout(
        barmode="group",
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font_color="white",
        legend_title_text="User",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3,
                    xanchor="center", x=0.5)
    )

    st.plotly_chart(fig, use_container_width=True)

# --- Activity Timeline with Annotated Latest Activities ---
def display_activity_chart(filtered_df):
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        graph_heading_with_info(
            "Tenant Activity Timeline",
            "Shows tenant activities over time. Toggle to show status (success/failure) info."
        )
    with col2:
        show_status = st.checkbox("Show Status", value=False)

    tenant_order = sorted(filtered_df["tenant"].unique())

    if show_status:
        filtered_df["status"] = filtered_df["status"].apply(lambda x: "success" if x != "failed" and x else "failed")
        color_map = {"success": "green", "failed": "red"}

        fig = px.scatter(
            filtered_df,
            x="ts",
            y="tenant",
            color="status",
            color_discrete_map=color_map,
            hover_data=["username", "message", "type"],
            category_orders={"tenant": tenant_order},
            title="Tenant Activity with Success/Failure Status",
            height=500
        )
        fig.update_traces(marker=dict(size=12, symbol="circle"))

    else:
        fig = px.scatter(
            filtered_df,
            x="ts",
            y="tenant",
            color="tenant",
            hover_data=["message", "username"],
            category_orders={"tenant": tenant_order},
            title="Activity by Tenant Over Time",
            height=500
        )
        fig.update_traces(marker=dict(size=12))

        # Annotate latest action per tenant
    latest_activity = (
        filtered_df.sort_values("ts", ascending=False)
        .groupby("tenant")
        .first()
        .reset_index()
    )

    for _, row in latest_activity.iterrows():
        fig.add_annotation(
            x=row["ts"],
            y=row["tenant"],
            text=f"{row['type']}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#444",
            borderwidth=1,
            font=dict(size=12, color="black"),
        )

    st.plotly_chart(fig, use_container_width=True)


# --- Action Success vs Failure Timeline ---

# --- Weekly Activity Overview with Tenant Comparison ---
def display_weekly_activity(filtered_df):
    graph_heading_with_info(
        "Weekly Activity by Tenants",
        "Weekly count of all user activities grouped by tenant."
    )
    weeks_back = st.number_input("Showing data for past N weeks:", min_value=1, max_value=52, value=6, step=1)

    filtered_df["week_start"] = filtered_df["ts"] - pd.to_timedelta(filtered_df["ts"].dt.weekday, unit='D')
    filtered_df["week_start"] = filtered_df["week_start"].dt.date

    latest_week = filtered_df["week_start"].max()
    cutoff_week = latest_week - datetime.timedelta(weeks=weeks_back)
    df_recent = filtered_df[filtered_df["week_start"] >= cutoff_week]

    grouped = df_recent.groupby(["week_start", "tenant"]).size().reset_index(name="activity_count")
    pivot = grouped.pivot(index="week_start", columns="tenant", values="activity_count").fillna(0)
    pivot = pivot.sort_index()

    scatter_fig = px.scatter(
        filtered_df,
        x="ts",
        y="tenant",
        color="tenant"
    )
    color_map = {trace.name: trace.marker.color for trace in scatter_fig.data}

    fig = go.Figure()

    for tenant in pivot.columns:
        fig.add_trace(go.Bar(
            x=pivot.index,
            y=pivot[tenant],
            name=tenant,
            marker_color=color_map.get(tenant, None),
            hovertemplate="Week: %{x}<br>Tenant: <b>" + tenant + "</b><br>Count: %{y}<extra></extra>",
        ))

    fig.update_layout(
        barmode='group',
        title=f"Weekly Activity Count (Last {weeks_back} Weeks)",
        xaxis_title="Week Starting",
        yaxis_title="Activity Count",
        xaxis=dict(type='category', tickangle=-45),
        height=600,
        legend_title_text="Tenant",
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font_color="white"
    )

    st.plotly_chart(fig, use_container_width=True)
    return pivot

# --- Daily activity chart ---
def display_daily_activity(pivot, filtered_df, selected_tenants):
    available_weeks = pivot.index.astype(str).tolist()
    graph_heading_with_info(
        "View Daily Activity Across Tenants",
        "Displays tenant-wise activity distributed across weekdays. Useful for workload patterns."
    )
    select_all = st.checkbox("Select All Weeks", value=True)

    if select_all:
        selected_weeks = available_weeks
    else:
        selected_weeks = st.multiselect("Select Week(s)", available_weeks, default=available_weeks[-1:])

    combined_daywise = pd.DataFrame()

    for selected_week in selected_weeks:
        start_week = datetime.datetime.strptime(selected_week, "%Y-%m-%d").date()
        end_week = start_week + datetime.timedelta(days=7)

        df_week = filtered_df[
            (filtered_df["tenant"].isin(selected_tenants)) &
            (filtered_df["date"] >= start_week) &
            (filtered_df["date"] < end_week)
        ].copy()

        df_week["weekday"] = pd.to_datetime(df_week["date"]).dt.day_name()
        df_week["week"] = selected_week

        combined_daywise = pd.concat([combined_daywise, df_week])

    if not combined_daywise.empty:
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        all_pairs = pd.DataFrame(
            list(product(weekday_order, selected_tenants)),
            columns=["weekday", "tenant"]
        )

        grouped = (
            combined_daywise.groupby(["weekday", "tenant"])
            .size()
            .reset_index(name="count")
        )

        # --- Merge actual counts with all weekday–tenant pairs ---
        merged = pd.merge(all_pairs, grouped, on=["weekday", "tenant"], how="left").fillna(0)
        merged["count"] = merged["count"].astype(int)

        fig_combined = px.bar(
            merged,
            x="weekday",
            y="count",
            color="tenant",
            barmode="group",
            category_orders={"weekday": weekday_order},
            title=f"Weekday Activity for Selected Week(s): {', '.join(selected_weeks)}",
            labels={"count": "Activity Count", "weekday": "Weekday", "tenant": "Tenant"},
            height=500
        )
        fig_combined.update_layout(
            plot_bgcolor="#111111",
            paper_bgcolor="#111111",
            font_color="white"
        )
        st.plotly_chart(fig_combined, use_container_width=True)

    else:
        st.info("No daily activity data for selected tenants and weeks.")
    return combined_daywise, selected_weeks, available_weeks, select_all

# --- Top N Active Users per Tenant ---
def display_top_active_users(combined_daywise):
    graph_heading_with_info(
        "Top N Active Users per Tenant",
        "Ranks users based on number of activities performed within selected weeks."
    )
    top_n = st.number_input("Select how many top users to show per tenant:", min_value=1, max_value=10, value=2, step=1)

    user_activity = (
        combined_daywise.groupby(["tenant", "username"])
        .size()
        .reset_index(name="action_count")
    )

    top_users = (
        user_activity.sort_values(["tenant", "action_count"], ascending=[True, False])
        .groupby("tenant")
        .head(top_n)
    )

    for tenant in top_users["tenant"].unique():
        tenant_users = top_users[top_users["tenant"] == tenant]
        with st.expander(f"Tenant: {tenant}"):
            for _, row in tenant_users.iterrows():
                key = f"expand_{tenant}_{row['username']}"
                if st.button(f"👤 {row['username']} ({row['action_count']} activities)", key=key + "_btn"):
                    st.session_state[key] = not st.session_state.get(key, False)
                if st.session_state.get(key, False):
                    user_df = combined_daywise[
                        (combined_daywise["tenant"] == tenant) & 
                        (combined_daywise["username"] == row["username"])
                    ]

                    user_df["weekday"] = user_df["ts"].dt.day_name()

                    weekday_counts = (
                        user_df.groupby("weekday")
                        .size()
                        .reset_index(name="count")
                    )

                    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    full_weekdays = pd.DataFrame({"weekday": weekday_order})
                    full_counts = pd.merge(full_weekdays, weekday_counts, on="weekday", how="left").fillna(0)
                    full_counts["count"] = full_counts["count"].astype(int)

                    fig_user = px.bar(
                        full_counts,
                        x="weekday",
                        y="count",
                        title=f"Weekday Activity for {row['username']} in {tenant}",
                        category_orders={"weekday": weekday_order},
                        labels={"count": "Action Count", "weekday": "Weekday"},
                        height=400
                    )
                    fig_user.update_layout(
                        plot_bgcolor="#111111",
                        paper_bgcolor="#111111",
                        font_color="white"
                    )
                    st.plotly_chart(fig_user, use_container_width=True)

# --- Instance Type Distribution ---
def instance_type_distribution(selected_tenants, date_range, selected_weeks, available_weeks, select_all):
    instance_df = fetch_instance_counts()

    if instance_df.empty:
        st.warning("No instance action data found.")
        return

    instance_df["action_ts"] = pd.to_datetime(instance_df["action_ts"])
    instance_df["date"] = instance_df["action_ts"].dt.date
    instance_df["week_start"] = instance_df["action_ts"] - pd.to_timedelta(instance_df["action_ts"].dt.weekday, unit="D")
    instance_df["week_start"] = instance_df["week_start"].dt.date.astype(str)
    instance_df["action_type"] = instance_df["action_type"].str.upper()
    instance_df = instance_df[instance_df["tenant"].isin(selected_tenants)]

    week_filter = selected_weeks if not select_all else available_weeks

    provisioned_df = instance_df[
        (instance_df["action_type"] == "PROVISIONS") &
        (instance_df["date"] >= date_range[0]) &
        (instance_df["date"] <= date_range[1])
    ]

    deleted_df = instance_df[
        (instance_df["action_type"] == "DELETES") &
        (instance_df["date"] >= date_range[0]) &
        (instance_df["date"] <= date_range[1])
    ]

    cumulative_df = provisioned_df[~provisioned_df["instance_id"].isin(deleted_df["instance_id"])]

    weekly_df = instance_df[
        (instance_df["week_start"].isin(week_filter)) &
        (instance_df["date"] >= date_range[0]) &
        (instance_df["date"] <= date_range[1])
    ]

    graph_heading_with_info(
        "Total Instances per Tenant",
        "This table shows the cumulative instances for the selected date range.<br>"
        "<span style='color:limegreen;'>+N</span>: Instances provisioned in selected weeks.<br>"
        "<span style='color:red;'>-N</span>: Instances deleted in selected weeks."
    )

    # --- Summary Table ---
    summary_data = []
    for tenant in selected_tenants:
        total_instances = cumulative_df[cumulative_df["tenant"] == tenant]["instance_id"].nunique()
        tenant_week_df = weekly_df[weekly_df["tenant"] == tenant]
        provisioned = tenant_week_df[tenant_week_df["action_type"] == "PROVISIONS"]["instance_id"].nunique()
        deleted = tenant_week_df[tenant_week_df["action_type"] == "DELETES"]["instance_id"].nunique()

        formatted = (
            f"{total_instances}  "
            f"<span style='color:limegreen'>+{provisioned}</span> "
            f"<span style='color:red'>-{deleted}</span>"
        )

        summary_data.append({"Tenant": tenant, "Total Instances": formatted})

    styled_df = pd.DataFrame(summary_data)
    table_html = styled_df.to_html(
        escape=False,
        index=False,
        classes="styled-instance-table"
    )

    st.markdown("""
        <style>
            .styled-instance-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 16px;
            }
            .styled-instance-table th, .styled-instance-table td {
                border: 1px solid #444;
                padding: 10px 16px;
                text-align: left;
            }
            .styled-instance-table th {
                background-color: #222;
                color: white;
            }
            .styled-instance-table tr:nth-child(even) {
                background-color: #1e1e1e;
            }
            .styled-instance-table tr:nth-child(odd) {
                background-color: #111111;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(table_html, unsafe_allow_html=True)

    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        graph_heading_with_info(
            "Filtered Instance Type Distribution",
            "Breakdown of instance types provisioned or deleted for each tenant during the selected timeframe. Toggle to include/exclude deleted instances."
        )
    with col2:
        show_deleted = st.checkbox("Include Deleted", value=False)

    deleted_ids = set(deleted_df["instance_id"])
    filtered_prov_df = provisioned_df[~provisioned_df["instance_id"].isin(deleted_ids)].copy()
    filtered_prov_df["status"] = "Provisioned"

    if show_deleted:
        filtered_del_df = deleted_df.copy()
        filtered_del_df["status"] = "Deleted"
        combined_df = pd.concat([filtered_prov_df, filtered_del_df], ignore_index=True)
    else:
        combined_df = filtered_prov_df

    instance_type_counts = (
        combined_df
        .groupby(["tenant", "instance_type", "status"])["instance_id"]
        .nunique()
        .reset_index(name="count")
    )

    view_mode = st.radio("Select View Mode:", ["Absolute Count", "Percentage"])
    if view_mode == "Percentage":
        total_per_group = instance_type_counts.groupby(["tenant", "status"])["count"].transform("sum")
        instance_type_counts["percent"] = (instance_type_counts["count"] / total_per_group * 100).round(2)
        y_col = "percent"
    else:
        y_col = "count"
    
    fig_type = px.bar(
        instance_type_counts,
        x="instance_type",
        y=y_col,
        color="tenant",
        pattern_shape="status" if show_deleted else None,
        pattern_shape_sequence=["", "/"],
        barmode="group",
        labels={
            "instance_type": "Instance Type",
            "tenant": "Tenant",
            "count": "Instance Count",
            "percent": "Percentage",
            "status": "Status"
        },
        title=f"Instance Types per Tenant ({'with Deleted' if show_deleted else 'Provisioned Only'})",
        height=500
    )

    fig_type.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font_color="white"
    )

    st.plotly_chart(fig_type, use_container_width=True)
    # --- Plan Distribution ---
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        graph_heading_with_info(
            "Filtered Machine Type Distribution",
            "Breakdown of instance plans provisioned or deleted for each tenant. Toggle to include/exclude deleted plans."
        )
    with col2:
        show_deleted = st.checkbox("Include Deleted", value=False, key="plan_deleted_checkbox")

        deleted_ids = set(deleted_df["instance_id"])
    filtered_prov_df = provisioned_df[~provisioned_df["instance_id"].isin(deleted_ids)].copy()
    filtered_prov_df["status"] = "Provisioned"

    if show_deleted:
        filtered_del_df = deleted_df.copy()
        filtered_del_df["status"] = "Deleted"
        combined_df = pd.concat([filtered_prov_df, filtered_del_df], ignore_index=True)
    else:
        combined_df = filtered_prov_df

    plan_counts = (
        combined_df
        .groupby(["tenant", "instance_plan", "status"])["instance_id"]
        .nunique()
        .reset_index(name="count")
    )

    view_mode = st.radio("Plan View Mode:", ["Absolute Count", "Percentage"], key="plan_view_mode")
    if view_mode == "Percentage":
        total_per_group = plan_counts.groupby(["tenant", "status"])["count"].transform("sum")
        plan_counts["percent"] = (plan_counts["count"] / total_per_group * 100).round(2)
        y_col = "percent"
    else:
        y_col = "count"

    fig = px.bar(
        plan_counts,
        x="instance_plan",
        y=y_col,
        color="tenant",
        pattern_shape="status" if show_deleted else None,
        pattern_shape_sequence=["", "/"],
        barmode="group",
        labels={
            "instance_plan": "Instance Plan",
            "tenant": "Tenant",
            "count": "Machine Type count",
            "percent": "Percentage",
            "status": "Status"
        },
        title=f"Instance Plans per Tenant ({'with Deleted' if show_deleted else 'Provisioned Only'})",
        height=500
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font_color="white"
    )

    st.plotly_chart(fig, use_container_width=True)

# --- Gantt Chart ---
def display_tenant_gantt_chart(selected_tenants, date_range, selected_weeks, available_weeks, select_all):

    graph_heading_with_info(
        "Tenant-Level Gantt Chart of Runs (Avg CPU %)",
        "Visualizes weekly run durations per tenant and their average CPU usage."
    )

    runs_df = fetch_run_data()
    instance_df = fetch_instance_counts()

    if not runs_df.empty:
        runs_df = runs_df[runs_df["tenant"].isin(selected_tenants)].copy()
        runs_df["week_start"] = runs_df["start"] - pd.to_timedelta(runs_df["start"].dt.weekday, unit="D")
        runs_df["week_start"] = runs_df["week_start"].dt.date.astype(str)

        week_filter = selected_weeks if not select_all else available_weeks
        filtered_runs = runs_df[
            (runs_df["week_start"].isin(week_filter)) &
            (
                ((runs_df["start"].dt.date >= date_range[0]) & (runs_df["start"].dt.date <= date_range[1])) |
                ((runs_df["end"].dt.date >= date_range[0]) & (runs_df["start"].dt.date <= date_range[1]))
            )
        ].copy()

        if not filtered_runs.empty:
            tenant_instance_counts = (
                instance_df.groupby("tenant")["instance_id"]
                .nunique()
                .reset_index()
                .rename(columns={"instance_id": "total_instances"})
            )

            cpu_summary = (
                filtered_runs.groupby(["tenant", "week_start"])
                .agg(
                    total_cpu=("avg_cpu", "sum"),
                    start=("start", "min"),
                    end=("end", "max"),
                    run_count=("avg_cpu", "count"),
                    running_instances=("instance_id", pd.Series.nunique)
                )
                .reset_index()
            )

            tenant_summary = pd.merge(cpu_summary, tenant_instance_counts, on="tenant", how="left")
            tenant_summary["avg_cpu"] = (
                tenant_summary["total_cpu"] / tenant_summary["run_count"]
            ).fillna(0).round(2)

            fig = px.timeline(
                tenant_summary,
                x_start="start",
                x_end="end",
                y="tenant",
                color="avg_cpu",
                color_continuous_scale="Viridis",
                labels={"avg_cpu": "Avg CPU (%)"},
                title="Tenant Run Activity Timeline (Avg CPU % per Run per Week)",
                hover_data={
                    "start": True,
                    "end": True,
                    "avg_cpu": True,
                    "total_instances": True,
                    "running_instances": True
                }
            )

            fig.update_yaxes(autorange="reversed")
            fig.update_layout(
                height=600,
                plot_bgcolor="#111111",
                paper_bgcolor="#111111",
                font_color="white",
                coloraxis_colorbar=dict(title="Avg CPU %")
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No run data available for selected filters.")

# --- Insights Section ---
def insights(selected_tenants):
    graph_heading_with_info(
        "Tenant-wise Temporal Behavior Insights (AI Generated)",
        "These insights are derived from deepseek-chat-v3-0324 model analyzing temporal activity across tenants, highlighting anomalies, patterns, and inefficiencies."
    )
    with st.expander("View AI Insight for selected Tenant(s)", expanded=True):
        tenants = selected_tenants if selected_tenants else None
        temp_df = fetch_temporal_activity_data(tenants)

        if temp_df.empty:
            st.warning("No temporal activity data found for the selected tenants.")
            return

        tenant_payloads = prepare_llm_friendly_json(temp_df)

        for tenant_name, payload in tenant_payloads.items():
            payload_json = json.dumps(payload, indent=2)

            with st.spinner(f"Analyzing behavior for {tenant_name}..."):
                insight = get_temporal_insights_from_ai(payload_json)

            with st.expander(f"🔹 {tenant_name} Insights", expanded=True):
                # Use container with scrolling for long insights
                st.markdown(f"""
                    <div style="
                        max-height: 300px; 
                        overflow-y: auto; 
                        padding: 10px; 
                        background: #1e1e1e; 
                        border-radius: 5px;
                        border: 1px solid #444;
                    ">
                        {insight}
                    </div>
                """, unsafe_allow_html=True)

# --- BCG Matrix ---
def display_bcg_matrix():
    graph_heading_with_info(
        "BCG Matrix: User VM Provisioning vs CPU Usage",
        "Shows user behavior in terms of VM provisioning vs average CPU utilization."
    )

    inst_df = fetch_instance_counts()
    run_df = fetch_run_data()

    if inst_df.empty:
        st.warning("No user or instance action data available.")
        return

    all_users = inst_df[["tenant", "username"]].drop_duplicates()

    provs = inst_df[inst_df["action_type"] == "PROVISIONS"]
    provs = provs[["tenant", "username", "instance_id"]].dropna().drop_duplicates()
    provision_counts = (
        provs.groupby(["tenant", "username"])
        .agg(num_vms=("instance_id", "nunique"))
        .reset_index()
    )

    run_df["start"] = pd.to_datetime(run_df["start"])
    run_df = run_df[["tenant", "instance_id", "avg_cpu"]]

    merged = pd.merge(provs, run_df, on=["tenant", "instance_id"], how="left")

    cpu_usage = (
        merged.groupby(["tenant", "username"])
        .agg(avg_cpu=("avg_cpu", "mean"))
        .reset_index()
    )

    user_stats = pd.merge(all_users, provision_counts, on=["tenant", "username"], how="left")
    user_stats = pd.merge(user_stats, cpu_usage, on=["tenant", "username"], how="left")

    user_stats["num_vms"] = user_stats["num_vms"].fillna(0).astype(int)
    user_stats["avg_cpu"] = user_stats["avg_cpu"].fillna(0)

    if user_stats.empty:
        st.info("No user activity found.")
        return
    
    fig = px.scatter(
        user_stats,
        x="num_vms",
        y="avg_cpu",
        color="tenant",
        hover_data=["username", "num_vms", "avg_cpu"],
        labels={
            "num_vms": "VMs Provisioned",
            "avg_cpu": "Avg CPU Usage (%)"
        },
        title="BCG Matrix: Users by VM Count and Avg CPU Usage",
        height=600
    )
    fig.update_traces(marker=dict(size=12))
    x_median = user_stats["num_vms"].median()
    y_median = user_stats["avg_cpu"].median()

    fig.add_shape(type="line", x0=x_median, x1=x_median, y0=0, y1=user_stats["avg_cpu"].max(),
                  line=dict(dash="dot", color="white"))
    fig.add_shape(type="line", x0=0, x1=user_stats["num_vms"].max(), y0=y_median, y1=y_median,
                  line=dict(dash="dot", color="white"))

    fig.update_layout(
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font_color="white"
    )

    st.plotly_chart(fig, use_container_width=True)

def chatbot_ui():
    with st.expander("💬 Chatbot - Ask about tenant patterns & anomalies", expanded=True):
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_query = st.chat_input("Ask your question here...")
        if user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                waiting_placeholder = st.empty()
                waiting_placeholder.markdown("*Searching for insights...*")

                try:
                    answer = rag_engine.rag_hybrid(user_query)  # or rag_summary_only, or normal_chat
                except Exception as e:
                    answer = f"Error: {e}"

                waiting_placeholder.markdown(answer)

            st.session_state.chat_history.append({"role": "assistant", "content": answer})