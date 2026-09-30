using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

// Entry point for headless data generation (generate_table3_data.sh):
//   Unity -batchmode -projectPath <copy> -executeMethod Table3DataGen.Run
// Opens the simulation scene and enters play mode; the patched
// Simulation.cs exits the editor once T3_TARGET_LINES training lines exist.
public static class Table3DataGen
{
    public static void Run()
    {
        EditorSceneManager.OpenScene("Assets/Scenes/SampleScene.unity");
        Debug.Log("T3: entering play mode");
        EditorApplication.isPlaying = true;
    }
}
