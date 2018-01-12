import requests
from bs4 import BeautifulSoup
import pandas as pd

teams = ['car','nyg','htx','tam','mia','crd','cin','kan','jax',
		 'rai','nwe','was','dal','nyj','rav','clt','gnb','oti',
 		 'den','min','chi','atl','nor','sea','ram','sdg',
 	  	 'sfo','det','cle','buf','phi','pit']

csv_teams = ['CAR','NYG','HOU','TB','MIA','ARI','CIN','KC','JAX',
 			 'OAK','NE','WAS','DAL','NYJ','BAL','IND','GB','TEN',
 			 'DEN','MIN','CHI','ATL','NO','SEA','LA','LAC',
 			 'SF','DET','CLE','BUF','PHI','PIT']

years = [str(i) for i in range(2013, 2018)]

df = pd.read_csv('../data/combined_data.csv')
off_df = pd.DataFrame(columns=['SeasonYear', 'OffenseTeam', 'Coach', 'Offensive_coordinator', 'Offensive_scheme'])
def_df = pd.DataFrame(columns=['SeasonYear', 'DefenseTeam', 'Defensive_coordinator', 'Defensive_alignment'])
team_dict = {}
for x, y in zip(teams, csv_teams):
	team_dict[x] = y

for year in years:
	for team in teams:
		url = 'https://www.pro-football-reference.com/teams/{}/{}.htm'.format(team, year)
		response = requests.get(url)
		content = response.content
		soup = BeautifulSoup(content, "html.parser")
		coach_soup = soup.find('div', {'id':'info', 'class':'teams'})
		p_tags = coach_soup.findAll('p')

		off_results = {'SeasonYear': floatyear, 'OffenseTeam': team_dict[team]}
		def_results = {'SeasonYear': year, 'DefenseTeam': team_dict[team]}
		for tag in p_tags:
			tag_text = tag.text
			if 'Coach' in tag_text:
				coach = tag_text.split(':')[1].split('(')[0].strip()
				off_results['Coach'] = coach
			if 'Offensive Coordinator' in tag_text:
				offensive_coordinator = tag_text.split(':')[1].strip()
				off_results['Offensive_coordinator'] = offensive_coordinator
			if 'Offensive Scheme' in tag_text:
				offensive_scheme = tag_text.split(':')[1].strip()
				off_results['Offensive_scheme'] = offensive_scheme
			if 'Defensive Coordinator' in tag_text:
				defensive_coordinator = tag_text.split(':')[1].strip()
				def_results['Defensive_coordinator'] = defensive_coordinator
			if 'Defensive Alignment' in tag_text:
				defensive_alignment = tag_text.split(':')[1].strip()
				def_results['Defensive_alignment'] = defensive_alignment

		off_df = off_df.append(off_results, ignore_index=True)
		def_df = def_df.append(def_results, ignore_index=True)

off_df.to_csv('../data/offensive_coach_data.csv', index=False)
def_df.to_csv('../data/defensive_coach_data.csv', index=False)

off_df['SeasonYear'] = off_df['SeasonYear'].astype(float)
def_df['SeasonYear'] = off_df['SeasonYear'].astype(float)
combined_df = pd.merge(df, off_df, how='left', left_on=['SeasonYear', 'OffenseTeam'], right_on=['SeasonYear', 'OffenseTeam'])
combined_df = pd.merge(combined_df, def_df, how='left', left_on=['SeasonYear', 'DefenseTeam'], right_on=['SeasonYear', 'DefenseTeam'])




