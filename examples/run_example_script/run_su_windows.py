
import time

from i_aoma.helper_ssicov_timeout import esegui





t0 = time.time()


results = esegui(stringa='esecuzione lunga', seconds=10)


t1 = time.time()
print(t1-t0,' elapsed seconds')





print('sono qui')


