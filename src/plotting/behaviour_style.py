BEHAVIOUR_COLOURS = {
  'speed': '#a4c2f4ff', # allen speed
  'orientation_corr': '#674ea7',
  'eye_pos': '#5691a5ff',
  'all_behav': '#5691A5',
  'hand_pos': '#93c47d',
  'hand_vel': '#a559aa',
  'Speed': '#93c47d', # rat in a box speed
  'Position': '#5691A5',
  'Head Direction': '#A559AA',
}

def get_behaviour_colour(behaviour):
  return BEHAVIOUR_COLOURS[behaviour]

def behaviours_to_colours(behaviours):
  return list(map(get_behaviour_colour, behaviours))

BEHAVIOUR_DISP_NAME= {
  'speed': 'Speed', # allen speed
  'orientation_corr': 'Orientation',
  'eye_pos': 'Eye Position',
  'hand_pos': 'Hand Position',
  'hand_vel': 'Hand Velocity',
  'Speed': 'Speed', # rat in a box speed
  'Position': 'Position',
  'all_behav': 'All Behaviours',
  'Head Direction': 'Head Direction',
  'head_direction': 'Head Direction',
  'pos': 'Position',
  'unknown': 'Residual',
  'orientation': "Orientation"
}
def get_disp_name(behaviour):
  return BEHAVIOUR_DISP_NAME[behaviour]
