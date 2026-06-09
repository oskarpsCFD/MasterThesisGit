#!/bin/sh
cd "${0%/*}" || exit 1
. "${WM_PROJECT_DIR:?}/bin/tools/RunFunctions"

application="overRhoPimpleDyMFoam"
refinementDict="system/rRefinementDict"

if [ -z "$1" ]; then
    echo "Usage: $0 <neumann_iterations>"
    exit 1
fi
NEUMANN_ITERATIONS="$1"

normalSteps=29
neumannSteps=1
pairSteps=$((normalSteps + neumannSteps))

# TOTAL NUMBER OF RUNS
totalCycles=60

# Read latest time
currentTime=$(foamListTimes -latestTime 2>/dev/null | tail -1)
currentTime=${currentTime:-0}

echo "Starting from t=$currentTime"

foamDictionary -entry customRefinement/refinement -set "true" "$refinementDict"

for i in $(seq 1 $totalCycles)
do
    phase=$(( currentTime % pairSteps ))

    if [ "$phase" -lt "$normalSteps" ]; then
        # NORMAL STEP
        sed -i "s/^\(NEUMANN_ITERATIONS =\s*\)[0-9]*/\10/" RefineMesh.py
        echo "=== Run $i / $totalCycles : t=$currentTime -> t=$((currentTime + 1)) (NEUMANN=0) ==="

    else
        # NEUMANN STEP
        sed -i "s/^\(NEUMANN_ITERATIONS =\s*\)[0-9]*/\1${NEUMANN_ITERATIONS}/" RefineMesh.py
        echo "=== Run $i / $totalCycles : t=$currentTime -> t=$((currentTime + 1)) (NEUMANN=$NEUMANN_ITERATIONS) ==="
    fi

    nextTime=$((currentTime + 1))

    foamDictionary system/controlDict -entry startTime -set $currentTime
    foamDictionary system/controlDict -entry endTime   -set $nextTime

    $application

    currentTime=$nextTime
done

# Reset afterwards
sed -i "s/^\(NEUMANN_ITERATIONS =\s*\)[0-9]*/\10/" RefineMesh.py

echo "Completed $totalCycles runs."
