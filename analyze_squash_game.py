import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
import json
import argparse
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def load_ball_data(ball_tracking_dir):
    """Load all ball tracking data from the given directory."""
    ball_tracking_dir = Path(ball_tracking_dir)
    
    # Load ball trajectory data
    ball_trajectory = np.load(ball_tracking_dir / "ball_trajectory.npy", allow_pickle=True)
    
    # Load ball positions CSV
    ball_positions_df = pd.read_csv(ball_tracking_dir / "ball_positions.csv")
    
    return {
        "trajectory": ball_trajectory,
        "positions_df": ball_positions_df,
        "dir": ball_tracking_dir
    }

def load_player_data(player_tracking_dir):
    """Load all player tracking data from the given directory."""
    player_tracking_dir = Path(player_tracking_dir)
    
    # Load player tracking CSV
    player_df = pd.read_csv(player_tracking_dir / "player_tracking_data.csv")
    
    return {
        "player_df": player_df,
        "dir": player_tracking_dir
    }

def analyze_ball_data(ball_data):
    """Analyze ball trajectory and movement patterns."""
    positions_df = ball_data["positions_df"]
    
    # Basic statistics
    x_mean = positions_df['x'].mean()
    y_mean = positions_df['y'].mean()
    x_std = positions_df['x'].std()
    y_std = positions_df['y'].std()
    
    # Calculate velocity and acceleration if timestamps available
    if 'timestamp' in positions_df.columns:
        positions_df = positions_df.sort_values('timestamp')
        positions_df['x_diff'] = positions_df['x'].diff()
        positions_df['y_diff'] = positions_df['y'].diff()
        positions_df['time_diff'] = positions_df['timestamp'].diff()
        positions_df['velocity_x'] = positions_df['x_diff'] / positions_df['time_diff']
        positions_df['velocity_y'] = positions_df['y_diff'] / positions_df['time_diff']
        positions_df['velocity'] = np.sqrt(positions_df['velocity_x']**2 + positions_df['velocity_y']**2)
        positions_df['acceleration_x'] = positions_df['velocity_x'].diff() / positions_df['time_diff']
        positions_df['acceleration_y'] = positions_df['velocity_y'].diff() / positions_df['time_diff']
        positions_df['acceleration'] = np.sqrt(positions_df['acceleration_x']**2 + positions_df['acceleration_y']**2)
    
    # Court regions analysis
    # Assuming court coordinates are normalized between 0-1 or are in pixels
    positions_df['court_region'] = pd.cut(positions_df['y'], bins=3, labels=['Front', 'Middle', 'Back'])
    positions_df['court_side'] = pd.cut(positions_df['x'], bins=2, labels=['Left', 'Right'])
    region_count = positions_df.groupby(['court_region', 'court_side']).size().reset_index(name='count')
    
    return {
        "mean_position": (x_mean, y_mean),
        "position_std": (x_std, y_std),
        "processed_df": positions_df,
        "region_distribution": region_count,
        "court_time": positions_df.groupby('court_region').size().to_dict()
    }

def analyze_player_data(player_data):
    """Analyze player movement patterns and positioning."""
    df = player_data["player_df"]
    
    # Check for available columns
    print(f"Available columns in player data: {df.columns.tolist()}")
    
    # Check if we have player identification
    has_player_ids = 'player_id' in df.columns
    
    # Check if we have x/y coordinates or need to use different columns
    position_columns = []
    if 'x' in df.columns and 'y' in df.columns:
        x_col, y_col = 'x', 'y'
    else:
        # Try to find position columns with common names
        potential_x = ['x', 'X', 'pos_x', 'x_pos', 'x_position', 'player_x', 'position_x']
        potential_y = ['y', 'Y', 'pos_y', 'y_pos', 'y_position', 'player_y', 'position_y']
        
        x_col = next((col for col in potential_x if col in df.columns), None)
        y_col = next((col for col in potential_y if col in df.columns), None)
        
        # If we still don't have position columns, check for numeric columns
        if x_col is None or y_col is None:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) >= 2:
                print(f"Using numeric columns as position: {numeric_cols[:2]}")
                x_col, y_col = numeric_cols[:2]
            else:
                print("Warning: Could not identify position columns in player data")
                # Return empty results if we can't find position columns
                return {"status": "error", "message": "No position columns found in player data"}
    
    print(f"Using '{x_col}' and '{y_col}' as position columns")
    
    results = {}
    
    if has_player_ids:
        # Separate players
        player_ids = df['player_id'].unique()
        
        for player_id in player_ids:
            player_df = df[df['player_id'] == player_id]
            
            # Movement analysis
            player_movement = {
                "mean_position": (player_df[x_col].mean(), player_df[y_col].mean()),
                "std_position": (player_df[x_col].std(), player_df[y_col].std()),
                "court_coverage": player_df.groupby(['court_region']).size().to_dict() if 'court_region' in player_df.columns else None
            }
            
            # Calculate distance traveled if sequential frames
            if 'frame' in player_df.columns:
                player_df = player_df.sort_values('frame')
                player_df['x_diff'] = player_df[x_col].diff()
                player_df['y_diff'] = player_df[y_col].diff()
                player_df['movement'] = np.sqrt(player_df['x_diff']**2 + player_df['y_diff']**2)
                player_movement["total_distance"] = player_df['movement'].sum()
                player_movement["avg_speed"] = player_df['movement'].mean()
            
            results[f"player_{player_id}"] = player_movement
    else:
        # Assume single player or treat all data as aggregate
        results["all_players"] = {
            "mean_position": (df[x_col].mean(), df[y_col].mean()),
            "std_position": (df[x_col].std(), df[y_col].std())
        }
        
        if 'frame' in df.columns:
            df = df.sort_values('frame')
            df['x_diff'] = df[x_col].diff()
            df['y_diff'] = df[y_col].diff()
            df['movement'] = np.sqrt(df['x_diff']**2 + df['y_diff']**2)
            results["all_players"]["total_distance"] = df['movement'].sum()
            results["all_players"]["avg_speed"] = df['movement'].mean()
    
    return results

def correlate_ball_player_data(ball_analysis, player_analysis):
    """Find correlations between ball and player movements."""
    correlations = {
        "player_ball_patterns": [],
        "strategic_insights": []
    }
    
    # Check if player tends to be near where the ball lands
    # This would require more detailed frame-by-frame analysis
    
    # Check court coverage vs ball distribution
    player_keys = player_analysis.keys()
    for player_key in player_keys:
        if "court_coverage" in player_analysis[player_key]:
            player_coverage = player_analysis[player_key]["court_coverage"]
            ball_coverage = ball_analysis["court_time"]
            
            # Simple correlation - which player follows ball pattern more closely
            # This is a placeholder for actual correlation calculation
            correlations["player_ball_patterns"].append({
                "player": player_key,
                "follows_ball": "Analysis would go here"
            })
    
    return correlations

def generate_visualizations(ball_data, ball_analysis, player_data, player_analysis, output_dir):
    """Generate visualizations of the analysis results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Ball position heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pd.crosstab(
            pd.cut(ball_data["positions_df"]['y'], bins=20),
            pd.cut(ball_data["positions_df"]['x'], bins=20)
        ),
        cmap='hot'
    )
    plt.title('Ball Position Heatmap')
    plt.savefig(output_dir / 'ball_heatmap_analysis.png')
    plt.close()
    
    # Ball trajectory over time
    if 'frame' in ball_data["positions_df"].columns:
        plt.figure(figsize=(12, 8))
        plt.plot(ball_data["positions_df"]['frame'], ball_data["positions_df"]['x'], label='X Position')
        plt.plot(ball_data["positions_df"]['frame'], ball_data["positions_df"]['y'], label='Y Position')
        plt.title('Ball Position Over Time')
        plt.xlabel('Frame')
        plt.ylabel('Position')
        plt.legend()
        plt.savefig(output_dir / 'ball_position_time.png')
        plt.close()
    
    # Player movement visualization
    if player_data and "player_df" in player_data:
        plt.figure(figsize=(10, 8))
        if 'player_id' in player_data["player_df"].columns:
            for player_id in player_data["player_df"]['player_id'].unique():
                player_df = player_data["player_df"][player_data["player_df"]['player_id'] == player_id]
                plt.scatter(player_df['x'], player_df['y'], alpha=0.5, label=f'Player {player_id}')
        else:
            plt.scatter(player_data["player_df"]['x'], player_data["player_df"]['y'], alpha=0.5)
        
        plt.title('Player Court Positions')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.savefig(output_dir / 'player_positions.png')
        plt.close()
    
    return output_dir

def generate_llm_feedback(ball_analysis, player_analysis, correlations, model_name=None):
    """Generate specialized feedback using a local LLM."""
    # If no model name is provided, return placeholder message
    if not model_name:
        return ("Model name not provided. To get LLM feedback, please provide a Hugging Face model name "
                "using the --hf_model_name parameter.")
    
    # Prepare the data for the LLM
    analysis_summary = {
        "ball_analysis": {
            "mean_position": ball_analysis["mean_position"],
            "position_std": ball_analysis["position_std"],
            "court_time": ball_analysis["court_time"],
            "region_distribution": ball_analysis["region_distribution"].to_dict() if hasattr(ball_analysis["region_distribution"], 'to_dict') else ball_analysis["region_distribution"]
        },
        "player_analysis": player_analysis,
        "correlations": correlations
    }
    
    # Convert the analysis to a readable format
    analysis_text = json.dumps(analysis_summary, default=str, indent=2)
    
    try:
        print(f"Loading model: {model_name}")
        
        # Create the prompt for the LLM
        prompt = f"""
        You are an expert squash coach and analyst. Analyze the following squash game data and provide specialized feedback:
        
        {analysis_text}
        
        Please provide:
        1. An overall assessment of the game pattern
        2. Specific strengths and weaknesses identified for each player
        3. Strategic recommendations for improvement
        4. Tactical insights based on the ball and player movement patterns
        5. Training drill recommendations to address the identified areas for improvement
        
        Focus on actionable insights and advanced squash technique and strategy.
        """
        
        # Initialize the model and tokenizer from HuggingFace
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load in 4-bit to reduce memory requirements for consumer GPUs
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto",  # Automatically use available GPUs
            load_in_4bit=True,  # Use 4-bit quantization to reduce memory
        )
        
        # Create a text generation pipeline
        text_generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=2000,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Generate text
        result = text_generator(prompt, return_full_text=False)
        feedback = result[0]["generated_text"]
        
        return feedback
    
    except Exception as e:
        return f"Error generating LLM feedback: {str(e)}\n\nRaw analysis data:\n{analysis_text}"

def export_results(ball_analysis, player_analysis, correlations, llm_feedback, output_dir):
    """Export all analysis results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Export numerical analysis
    analysis_results = {
        "ball_analysis": {k: v for k, v in ball_analysis.items() if k != "processed_df"},
        "player_analysis": player_analysis,
        "correlations": correlations
    }
    
    with open(output_dir / 'analysis_results.json', 'w') as f:
        json.dump(analysis_results, default=str, indent=2, fp=f)
    
    # Export LLM feedback
    with open(output_dir / 'coach_feedback.txt', 'w') as f:
        f.write(llm_feedback)
    
    # Export processed data
    if "processed_df" in ball_analysis:
        ball_analysis["processed_df"].to_csv(output_dir / 'processed_ball_data.csv', index=False)
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description='Analyze squash game tracking data and generate feedback.')
    parser.add_argument('--ball_tracking_dir', type=str, required=True, 
                        help='Directory containing ball tracking data')
    parser.add_argument('--player_tracking_dir', type=str, required=True, 
                        help='Directory containing player tracking data')
    parser.add_argument('--output_dir', type=str, default=f'analysis_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                        help='Directory to save analysis results')
    parser.add_argument('--hf_model_name', type=str, default=None,
                        help='Hugging Face model name for generating LLM feedback (e.g., "mistralai/Mistral-7B-Instruct-v0.2")')
    
    args = parser.parse_args()
    
    try:
        print(f"Loading ball tracking data from {args.ball_tracking_dir}")
        ball_data = load_ball_data(args.ball_tracking_dir)
        
        print(f"Loading player tracking data from {args.player_tracking_dir}")
        player_data = load_player_data(args.player_tracking_dir)
        
        print("Analyzing ball data...")
        ball_analysis = analyze_ball_data(ball_data)
        
        print("Analyzing player data...")
        player_analysis = analyze_player_data(player_data)
        
        if isinstance(player_analysis, dict) and "status" in player_analysis and player_analysis["status"] == "error":
            print(f"Error in player analysis: {player_analysis['message']}")
            # Continue with just ball analysis
            player_analysis = {"error": player_analysis["message"]}
        
        print("Correlating ball and player data...")
        correlations = correlate_ball_player_data(ball_analysis, player_analysis)
        
        print("Generating visualizations...")
        viz_dir = generate_visualizations(ball_data, ball_analysis, player_data, player_analysis, args.output_dir)
        
        print("Generating LLM feedback...")
        llm_feedback = generate_llm_feedback(ball_analysis, player_analysis, correlations, args.hf_model_name)
        
        print("Exporting results...")
        output_dir = export_results(ball_analysis, player_analysis, correlations, llm_feedback, args.output_dir)
        
        print(f"Analysis complete! Results saved to {output_dir}")
        print("\nCoach Feedback Preview:")
        print("="*80)
        print(llm_feedback[:500] + "..." if len(llm_feedback) > 500 else llm_feedback)
        print("="*80)
        print(f"Full feedback available in {output_dir / 'coach_feedback.txt'}")
    
    except Exception as e:
        import traceback
        print(f"Error during analysis: {str(e)}")
        traceback.print_exc()
        print("\nTips for troubleshooting:")
        print("1. Check the format of your CSV files - print the first few rows")
        print("2. Verify that ball_positions.csv has 'x' and 'y' columns")
        print("3. Check player_tracking_data.csv column names with:")
        print("   import pandas as pd")
        print("   df = pd.read_csv('tracking_output/player_tracking_20250327_163003/player_tracking_data.csv')")
        print("   print(df.columns.tolist())")

if __name__ == "__main__":
    main() 