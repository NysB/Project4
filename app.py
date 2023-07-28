# Import the dependencies.
import numpy as np

import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func
import pandas as pd

from flask import Flask, jsonify, render_template
from flask_cors import CORS, cross_origin

#################################################
# Database Setup
#################################################
engine = create_engine("sqlite:///NBA_stats.sqlite")

# reflect an existing database into a new model
Base = automap_base()
# reflect the tables
Base.prepare(engine, reflect=True)

# Save references to each table
Team_table= Base.classes.Team_table
Player_score_table = Base.classes.Player_score_table
Player_info_table = Base.classes.Player_info_table



#################################################
# Flask Setup
#################################################
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


#################################################
# Flask Routes
#################################################
@app.route("/")
def get_teamName():
    session = Session(engine)  
    res = session.query(Team_table.Team).all()
    session.close()

    team_names = [row[0] for row in res]
    return render_template('index.html', team_names=team_names)


# teams route
@app.route("/teams")
@cross_origin()
def get_teams():
    # Query the team table and return the results as JSON

    session = Session(engine)
    teams = session.query(Team_table.Team, Team_table.PPG, Team_table.APG, Team_table.RPG).all()
    session.close()
    
    results = []
    for Team, PPG, APG, RPG in teams:
        team_dict = {}
        team_dict["Team"] = Team
        team_dict["PPG"] = PPG
        team_dict["APG"] = APG
        team_dict["RPG"] = RPG
        results.append(team_dict)
    
    return jsonify(results)

#  player scores route
@app.route("/player_scores")
@cross_origin()
def get_player_scores():
    # Query the Player_score_table and return the results as JSON
    
    session = Session(engine)
    player_scores = session.query(Player_score_table.Player, Player_score_table.Year, Player_score_table.GP, Player_score_table.GS, Player_score_table.AST, Player_score_table.PTS, Player_score_table.TRB).all()
    session.close()

    results = []
    for Player, Year, GP, GS, AST,PTS, TRB in player_scores:
        player_score_dict = {}
        player_score_dict["Player"] = Player
        player_score_dict["Year"] = Year
        player_score_dict["GP"] = GP
        player_score_dict["GS"] = GS
        player_score_dict["AST"]= AST
        player_score_dict["PTS"]= PTS
        player_score_dict["TRB"]= TRB

        results.append(player_score_dict)
    
    return jsonify(results)


# player info route

@app.route("/player_info")
@cross_origin()
def get_player_info():
    # Query the player info table and return the results as JSON
    
    session = Session(engine)
    player_info = session.query(Player_info_table.Player, 
                                Player_info_table.Current_team, 
                                Player_info_table.Age, 
                                Player_info_table.Height,
                                Player_info_table.Weight,
                                Player_info_table.NBA_status,
                                Player_info_table.Draft_entry).all()
    session.close()

    results = []
    for Player, Current_team, Age, Height, Weight, NBA_status, Draft_entry in player_info:
        player_info_dict = {}
        player_info_dict["Player"] = Player
        player_info_dict["Current_team"] = Current_team
        player_info_dict["Age"] = Age
        player_info_dict["Height"] = Height
        player_info_dict["Weight"] = Weight
        player_info_dict["NBA Status"] = NBA_status
        player_info_dict["Draft Entry"] = Draft_entry
        results.append(player_info_dict)
    
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)