using UnityEngine;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer), typeof(MeshCollider))]
public class TerrainWithMaterial : MonoBehaviour
{
    [Header("Ustawienia Wyglądu")]
    public Material terrainMaterial;
    public string terrainTag = "ground";

    [Header("Ustawienia Obszaru")]
    public Vector2 centerPos = Vector2.zero;
    public float yOffset = 0f;
    public float width = 100f;
    public float length = 100f;
    public int xSegments = 10;
    public int zSegments = 10;
    public float height = 3.0f;

    [Header("Ustawienia Generacji")]
    public bool usePredictableSequence = true;
    public int seed = 12345;

    public void generateRandomTerrain()
    {
        if (usePredictableSequence)
        {
            Random.InitState(seed);
            seed++;
        }

        try
        {
            gameObject.tag = terrainTag;
        }
        catch (UnityException)
        {
            Debug.LogError($"[BŁĄD] Tag '{terrainTag}' nie istnieje! Dodaj go w ustawieniach Tags & Layers. Fizyka będzie działać, ale tag nie został ustawiony.");
        }

        Mesh mesh = GenerateMesh();

        GetComponent<MeshFilter>().mesh = mesh;

        MeshCollider mc = GetComponent<MeshCollider>();
        mc.sharedMesh = null; 
        mc.sharedMesh = mesh; 

        if (terrainMaterial != null)
        {
            GetComponent<MeshRenderer>().material = terrainMaterial;
        }
        else
        {
            Debug.LogWarning("Nie przypisano materiału w Inspektorze!");
        }
    }

    Mesh GenerateMesh()
    {
        Mesh mesh = new Mesh();

        float[,] heights = new float[xSegments + 1, zSegments + 1];
        for (int z = 0; z <= zSegments; z++)
        {
            for (int x = 0; x <= xSegments; x++)
            {
                heights[x, z] = yOffset + (Random.value * height);
            }
        }

        int numTiles = xSegments * zSegments;
        int numVertices = numTiles * 6;

        Vector3[] vertices = new Vector3[numVertices];
        int[] triangles = new int[numVertices];

        float startX = centerPos.x - width / 2;
        float startZ = centerPos.y - length / 2;
        float xStep = width / xSegments;
        float zStep = length / zSegments;

        int v = 0;

        for (int z = 0; z < zSegments; z++)
        {
            for (int x = 0; x < xSegments; x++)
            {
                float h_bl = heights[x, z];       
                float h_br = heights[x + 1, z];   
                float h_tl = heights[x, z + 1]; 
                float h_tr = heights[x + 1, z + 1];

                Vector3 bl = new Vector3(startX + x * xStep, h_bl, startZ + z * zStep);
                Vector3 br = new Vector3(startX + (x + 1) * xStep, h_br, startZ + z * zStep);
                Vector3 tl = new Vector3(startX + x * xStep, h_tl, startZ + (z + 1) * zStep);
                Vector3 tr = new Vector3(startX + (x + 1) * xStep, h_tr, startZ + (z + 1) * zStep);

                vertices[v] = bl; vertices[v + 1] = tl; vertices[v + 2] = br;
                vertices[v + 3] = tl; vertices[v + 4] = tr; vertices[v + 5] = br;

                for (int i = 0; i < 6; i++) triangles[v + i] = v + i;
                v += 6;
            }
        }

        mesh.vertices = vertices;
        mesh.triangles = triangles;

        mesh.RecalculateNormals();
        mesh.RecalculateBounds();

        return mesh;
    }
}
