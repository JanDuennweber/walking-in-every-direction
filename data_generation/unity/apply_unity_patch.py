#!/usr/bin/env python3
"""
Patches a COPY of Geyer's SensFloorSimulation Unity project so it can
generate the row 3/4 training data for Table 3 headless. Never run this on
the original project.

Changes, all driven by environment variables so that the unpatched
behavior is kept when a variable is not set:

Simulation.cs
  - T3_HEADINGS: CSV from generate_headings.py. Each walker's heading is
    read from it (absolute yaw in degrees, cycled) instead of the
    narrow-range sampler (entry-side base rotation + uniform [-60,60]).
    Walker count, entry position, MoCap prefab choice, standing people and
    noise are unchanged -- only the source of the heading changes
    (Subsection 3.2 of the paper).
  - Capture uses getInstanceMaskTrainingData(), the export that produced
    data/train_synthetic_old_range.txt (variable number of per-foot
    instance masks per line). The project was last left on
    getSemanticMaskTrainingData() (always 3 masks), a different format.
  - T3_TARGET_LINES: stop and exit the editor once the training file has
    at least this many lines with at least one foot instance mask. Lines
    without one (only noise triangles active) are still written but not
    counted; generate_table3_data.sh drops them, because
    train_synthetic_old_range.txt contains none (0 of 25918).
  - T3_TIMESCALE: Time.timeScale (default 1). Physics and animation run in
    game time, so a higher value only speeds up wall-clock time as long as
    the machine keeps up; keep 1 for results faithful to the original run.

FileWriteService.cs
  - T3_OUT_DIR: replaces the hardcoded Windows paths (D:\\18_Unity\\...).
    TrainingData.txt and the per-iteration message files go there.

Usage: apply_unity_patch.py <path to copied project>
"""

import sys
from pathlib import Path


def patch(path, replacements):
    with open(path, encoding="utf-8", newline="") as f:
        text = f.read()
    crlf = "\r\n" in text
    text = text.replace("\r\n", "\n")
    for old, new in replacements:
        # skip replacements already applied (resume on an existing copy):
        # the unique marker is the first line that exists only in `new`
        marker = next(l for l in new.split("\n") if l.strip() and l not in old)
        if marker in text:
            continue
        if text.count(old) != 1:
            sys.exit(f"{path.name}: expected exactly one match for:\n{old}")
        text = text.replace(old, new)
    if crlf:
        text = text.replace("\n", "\r\n")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print("patched", path)


REQUIRED = {
    "Simulation.cs": ["T3_HEADINGS", "t3TargetLines > 0 && FileWriteService.totalTrainingLines",
                      "TrainingDataWrapper data = getInstanceMaskTrainingData();",
                      "t3Headings[t3HeadingIndex % t3Headings.Count]", "t3Triangle != null",
                      "starting at index"],
    "FileWriteService.cs": ["T3_OUT_DIR", "totalTrainingLines++;", "resuming with"],
}


def check(scripts):
    for name, markers in REQUIRED.items():
        with open(scripts / name, encoding="utf-8", newline="") as f:
            text = f.read()
        missing = [m for m in markers if m not in text]
        if missing:
            sys.exit(f"{name}: patch incomplete, missing {missing}")
    print("all patch markers present")


def main():
    project = Path(sys.argv[1])
    scripts = project / "Assets" / "Scripts"

    patch(scripts / "Simulation.cs", [
        # config from environment
        ("""    void Start()
    {
        StartCoroutine(StartSimulation(500));
    }""",
         """    // Table 3 data generation (see experiments/sensfloor/data_generation/unity/)
    private static List<float> t3Headings = null;
    private static int t3HeadingIndex = 0;
    private static int t3TargetLines = 0;

    void Start()
    {
        string headingsPath = Environment.GetEnvironmentVariable("T3_HEADINGS");
        if (!string.IsNullOrEmpty(headingsPath))
        {
            t3Headings = new List<float>();
            foreach (string line in System.IO.File.ReadAllLines(headingsPath).Skip(1))
            {
                if (line.Trim().Length > 0)
                {
                    t3Headings.Add(float.Parse(line.Trim(), System.Globalization.CultureInfo.InvariantCulture));
                }
            }
            Debug.Log("T3: loaded " + t3Headings.Count + " headings from " + headingsPath);
        }
        string target = Environment.GetEnvironmentVariable("T3_TARGET_LINES");
        if (!string.IsNullOrEmpty(target))
        {
            t3TargetLines = Int32.Parse(target);
        }
        string timeScale = Environment.GetEnvironmentVariable("T3_TIMESCALE");
        if (!string.IsNullOrEmpty(timeScale))
        {
            Time.timeScale = float.Parse(timeScale, System.Globalization.CultureInfo.InvariantCulture);
        }
        StartCoroutine(StartSimulation(t3TargetLines > 0 ? Int32.MaxValue : 500));
    }"""),
        # stop once enough training lines are written
        ("""            OnIterationCompleted();
        }
        Debug.Log("All simulation iterations finished");""",
         """            OnIterationCompleted();
            if (t3TargetLines > 0 && FileWriteService.totalTrainingLines >= t3TargetLines)
            {
                Debug.Log("T3: reached " + FileWriteService.totalTrainingLines + " training lines, exiting");
#if UNITY_EDITOR
                UnityEditor.EditorApplication.Exit(0);
#else
                Application.Quit();
#endif
                yield break;
            }
        }
        Debug.Log("All simulation iterations finished");"""),
        # instance-mask export, as used for train_synthetic_old_range.txt
        ("""        //TrainingDataWrapper data = getInstanceMaskTrainingData();
        TrainingDataWrapper data = getSemanticMaskTrainingData();""",
         """        TrainingDataWrapper data = getInstanceMaskTrainingData();
        //TrainingDataWrapper data = getSemanticMaskTrainingData();"""),
        # heading source
        ("""            int rotationInDegreesAroundYAxis = rand.Next(0, 61);
            int sign = rand.Next(1, 3) == 1 ? 1 : -1;
            currWalker.rotation = baseRotation * UnityEngine.Quaternion.Euler(0, sign * rotationInDegreesAroundYAxis, 0);""",
         """            if (t3Headings != null)
            {
                float headingDeg = t3Headings[t3HeadingIndex % t3Headings.Count];
                t3HeadingIndex++;
                currWalker.rotation = UnityEngine.Quaternion.Euler(0, headingDeg, 0);
            }
            else
            {
                int rotationInDegreesAroundYAxis = rand.Next(0, 61);
                int sign = rand.Next(1, 3) == 1 ? 1 : -1;
                currWalker.rotation = baseRotation * UnityEngine.Quaternion.Euler(0, sign * rotationInDegreesAroundYAxis, 0);
            }"""),
    ])

    patch(scripts / "Simulation.cs", [
        # Geyer's cleanup assumed every collider a standing person's foot
        # touches is a floor triangle; a foot touching e.g. a walker's foot
        # made GetComponent return null and killed the simulation coroutine.
        ("""                        collidingTriangle.gameObject.GetComponent<PatchTriangleCollision>().active = false;""",
         """                        PatchTriangleCollision t3Triangle = collidingTriangle == null ? null : collidingTriangle.gameObject.GetComponent<PatchTriangleCollision>();
                        if (t3Triangle != null)
                        {
                            t3Triangle.active = false;
                        }"""),
        # on resume, continue at a random heading instead of repeating the file from its start
        ("""            Debug.Log("T3: loaded " + t3Headings.Count + " headings from " + headingsPath);""",
         """            t3HeadingIndex = new System.Random().Next(t3Headings.Count);
            Debug.Log("T3: loaded " + t3Headings.Count + " headings from " + headingsPath + ", starting at index " + t3HeadingIndex);"""),
    ])

    patch(scripts / "FileWriteService.cs", [
        ("""    private string destinationFolder = "D:\\\\18_Unity\\\\projects\\\\SensFloorSimulation\\\\Assets\\\\generatedData\\\\";
    private string trainingDestinationFolder = "D:\\\\18_Unity\\\\projects\\\\SensFloorSimulation\\\\Assets\\\\generatedData\\\\training\\\\";""",
         """    private string destinationFolder = "D:\\\\18_Unity\\\\projects\\\\SensFloorSimulation\\\\Assets\\\\generatedData\\\\";
    private string trainingDestinationFolder = "D:\\\\18_Unity\\\\projects\\\\SensFloorSimulation\\\\Assets\\\\generatedData\\\\training\\\\";
    public static int totalTrainingLines = 0;

    void Awake()
    {
        string outDir = Environment.GetEnvironmentVariable("T3_OUT_DIR");
        if (!string.IsNullOrEmpty(outDir))
        {
            destinationFolder = Path.Combine(outDir, "messages") + Path.DirectorySeparatorChar;
            trainingDestinationFolder = outDir + Path.DirectorySeparatorChar;
            Directory.CreateDirectory(destinationFolder);
        }
    }"""),
        # resume: count the lines with feet already in TrainingData.txt
        ("""            Directory.CreateDirectory(destinationFolder);
        }
    }""",
         """            Directory.CreateDirectory(destinationFolder);
            string existing = trainingDestinationFolder + "TrainingData.txt";
            if (File.Exists(existing))
            {
                foreach (string l in File.ReadLines(existing))
                {
                    if (l.Length > 0 && !l.EndsWith(":"))
                    {
                        totalTrainingLines++;
                    }
                }
                Debug.Log("T3: resuming with " + totalTrainingLines + " training lines already written");
            }
        }
    }"""),
        ("""                    writer.WriteLine(line);
                    writtenTrainingStrings++;""",
         """                    writer.WriteLine(line);
                    writtenTrainingStrings++;
                    if (!line.EndsWith(":"))
                    {
                        totalTrainingLines++;
                    }"""),
    ])

    check(scripts)
    editor_dir = project / "Assets" / "Editor"
    editor_dir.mkdir(exist_ok=True)
    src = Path(__file__).with_name("Table3DataGen.cs")
    (editor_dir / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    print("added", editor_dir / src.name)


if __name__ == "__main__":
    main()
