#!/bin/bash
# Light job feeder run on glacier: submits the lines of a job list (one "ENV=... sbatch ..." command
# per line, executed on mlsc from the semi repo) while keeping <= MAXJOBS of our sota_* jobs queued/running.
# Usage: MAXJOBS=5 scripts/sota/queue_jobs.sh <joblist.txt>
LIST=${1:?job list}; MAX=${MAXJOBS:-5}
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue; case "$cmd" in \#*) continue;; esac
  until [ "$(ssh -n -o BatchMode=yes mlsc 'squeue -h -u $USER -o %j' 2>/dev/null | grep -c '^sota_')" -lt "$MAX" ]; do sleep 120; done
  echo "$(date '+%F %T') submit: $cmd"
  ssh -n -o BatchMode=yes mlsc "cd /homes/8/dp003/code/development/semi && $cmd" 2>&1 | grep -v '^\*'
  sleep 20
done < "$LIST"
echo "$(date '+%F %T') all submitted"
