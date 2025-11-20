import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_top_10_states(df, output_dir='plots'):
    """Plots the top 10 registration states by number of violations."""
    states = df.groupby('Registration State')['Summons Number'].count()
    states = pd.DataFrame(states)
    states = states.sort_values(by='Summons Number', ascending=False)
    states.columns = ['Number of Violations']
    states['percentage'] = round(states['Number of Violations'] / states['Number of Violations'].sum() * 100, 2)
    top_10 = states.head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_10.index, y=top_10['percentage'])
    plt.title('Top 10 States by Number of Parking Violations')
    plt.xlabel('State')
    plt.ylabel('Percentage of Violations (%)')
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'top_10_states.png'))
        plt.close()
    else:
        plt.show()

def plot_top_10_violation_codes(df, output_dir='plots'):
    """Plots the top 10 violation codes by number of violations."""
    violation = df.groupby('Violation Code')['Summons Number'].count()
    violation = pd.DataFrame(violation)
    violation = violation.sort_values(by='Summons Number', ascending=False)
    violation.columns = ['Number of Violations']
    violation['percentage'] = round(violation['Number of Violations'] / violation['Number of Violations'].sum() * 100, 2)
    top_10 = violation.head(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_10.index, y=top_10['percentage'])
    plt.title('Top 10 Violation Codes by Number of Parking Violations')
    plt.xlabel('Violation Code')
    plt.ylabel('Percentage of Violations (%)')
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'top_10_violation_codes.png'))
        plt.close()
    else:
        plt.show()

def plot_violations_by_day_of_week(df, output_dir='plots'):
    """Plots the percentage of parking violations by day of the week."""
    day = df.groupby('Day of week')['Summons Number'].count()
    day = pd.DataFrame(day)
    day = day.sort_values(by='Summons Number', ascending=False)
    day.columns = ['Number of Violations']
    day['percentage'] = round(day['Number of Violations'] / day['Number of Violations'].sum() * 100, 2)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=day.index, y=day['percentage'])
    plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.title('Parking Violations by Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Percentage of All Violations (%)')
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'violations_by_day.png'))
        plt.close()
    else:
        plt.show()

def plot_violations_by_month(df, output_dir='plots'):
    """Plots the percentage of parking violations by month."""
    month = df.groupby('Month')['Summons Number'].count()
    month = pd.DataFrame(month)
    month = month.sort_values(by='Summons Number', ascending=False)
    month.columns = ['Number of Violations']
    month['percentage'] = round(month['Number of Violations'] / month['Number of Violations'].sum() * 100, 2)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=month.index, y=month['percentage'])
    plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.title('Parking Violations by Month')
    plt.xlabel('Month')
    plt.ylabel('Percentage of All Violations (%)')
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'violations_by_month.png'))
        plt.close()
    else:
        plt.show()

def plot_top_10_vehicle_makes(df, output_dir='plots'):
    """Plots the top 10 vehicle makes by number of violations."""
    vehicle_make = df.groupby('Vehicle Make')['Summons Number'].count()
    vehicle_make = pd.DataFrame(vehicle_make)
    vehicle_make = vehicle_make.sort_values(by='Summons Number', ascending=False)
    vehicle_make.columns = ['Number of Violations']
    vehicle_make['percentage'] = round(vehicle_make['Number of Violations'] / vehicle_make['Number of Violations'].sum() * 100, 2)
    top_10 = vehicle_make.head(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_10.index, y=top_10['percentage'])
    plt.title('Top 10 Vehicle Makes by Violations')
    plt.xlabel('Vehicle Make')
    plt.ylabel('Percentage of All Violations (%)')
    plt.xticks(rotation=45)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'top_10_vehicle_makes.png'))
        plt.close()
    else:
        plt.show()

def plot_top_10_streets(df, output_dir='plots'):
    """Plots the top 10 streets by number of violations."""
    street = df.groupby('Street Name')['Summons Number'].count()
    street = pd.DataFrame(street)
    street = street.sort_values(by='Summons Number', ascending=False)
    street.columns = ['Number of Violations']
    street['percentage'] = round(street['Number of Violations'] / street['Number of Violations'].sum() * 100, 2)
    top_10 = street.head(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_10.index, y=top_10['percentage'])
    plt.title('Top 10 Streets by Violations')
    plt.xlabel('Street Name')
    plt.ylabel('Percentage of All Violations (%)')
    plt.xticks(rotation=45, ha='right')
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'top_10_streets.png'))
        plt.close()
    else:
        plt.show()

def compare_ny_vs_non_ny(df, output_dir='plots'):
    """Compares violation codes for NY vs Non-NY registered vehicles."""
    ny_cars = df[df['Registration State'] == 'NY']
    non_ny_cars = df[df['Registration State'] != 'NY']
    
    ny_counts = ny_cars['Violation Code'].value_counts(normalize=True)
    non_ny_counts = non_ny_cars['Violation Code'].value_counts(normalize=True)
    
    comp = pd.DataFrame({
        'NY': ny_counts,
        'Non-NY': non_ny_counts
    }).fillna(0)
    comp = comp.sort_values(by='NY', ascending=False).head(10)
    
    comp.plot(kind='bar', figsize=(14, 7))
    plt.title('Top 10 Violation Codes: NY vs Non-NY')
    plt.xlabel('Violation Code')
    plt.ylabel('Proportion')
    plt.xticks(rotation=0)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'ny_vs_non_ny.png'))
        plt.close()
    else:
        plt.show()

def run_eda(df, output_dir='plots'):
    """Runs all EDA functions."""
    print("Generating EDA plots...")
    plot_violations_by_day_of_week(df, output_dir)
    plot_violations_by_month(df, output_dir)
    plot_top_10_vehicle_makes(df, output_dir)
    plot_top_10_streets(df, output_dir)
    plot_top_10_states(df, output_dir)
    plot_top_10_violation_codes(df, output_dir)
    compare_ny_vs_non_ny(df, output_dir)
    print(f"Plots saved to {output_dir}")
