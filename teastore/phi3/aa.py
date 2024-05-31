import os
import shutil

for rep in range(5):
    os.mkdir('llm_teastore/rep{}/generated_answers/'.format(rep))
    files = [a for a in os.listdir('llm_teastore/rep{}/generated_configs'.format(rep)) if '.txt' in a]
    for f in files:
        shutil.move('llm_teastore/rep{}/generated_configs/{}'.format(rep, f), 'llm_teastore/rep{}/generated_answers/{}'.format(rep, f))
