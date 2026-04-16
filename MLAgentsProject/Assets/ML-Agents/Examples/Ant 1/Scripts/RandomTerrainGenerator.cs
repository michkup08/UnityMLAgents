using UnityEngine;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer), typeof(MeshCollider))]
public class TerrainWithMaterial : MonoBehaviour
{
    [Header("Ustawienia Wyglądu")]
    public Material terrainMaterial;
    public string terrainTag = "ground"; // Upewnij się, że taki tag jest w Edit -> Project Settings -> Tags & Layers!

    [Header("Ustawienia Obszaru")]
    public Vector2 centerPos = Vector2.zero;
    public float width = 10f;
    public float length = 10f;
    public int xSegments = 10;
    public int zSegments = 10;

    public float height = 3.0f;

    public void generateRandomTerrain()
    {
        // 1. Bezpieczne zarządzanie Tagiem
        try
        {
            gameObject.tag = terrainTag;
        }
        catch (UnityException)
        {
            Debug.LogError($"[BŁĄD] Tag '{terrainTag}' nie istnieje! Dodaj go w ustawieniach Tags & Layers. Fizyka będzie działać, ale tag nie został ustawiony.");
        }

        // 2. Generowanie Geometrii
        Mesh mesh = GenerateMesh();

        // 3. Przypisanie Mesha
        GetComponent<MeshFilter>().mesh = mesh;

        // 4. Wymuszenie aktualizacji Collidera
        MeshCollider mc = GetComponent<MeshCollider>();
        mc.sharedMesh = null; // Czyszczenie starego stanu (wymusza odświeżenie)
        mc.sharedMesh = mesh; // Przypisanie nowej siatki

        // 5. Nałożenie materiału
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

        // KROK A: Wylosowanie wysokości dla wszystkich punktów przecięcia siatki
        float[,] heights = new float[xSegments + 1, zSegments + 1];
        for (int z = 0; z <= zSegments; z++)
        {
            for (int x = 0; x <= xSegments; x++)
            {
                heights[x, z] = Random.value * height; // Wysokość losowana od 0 do 1 * wysokość
            }
        }

        // KROK B: Przygotowanie tablic
        int numTiles = xSegments * zSegments;
        int numVertices = numTiles * 6;

        Vector3[] vertices = new Vector3[numVertices];
        int[] triangles = new int[numVertices];

        float startX = centerPos.x - width / 2;
        float startZ = centerPos.y - length / 2;
        float xStep = width / xSegments;
        float zStep = length / zSegments;

        int v = 0;

        // KROK C: Budowanie trójkątów na podstawie zapisanych wysokości
        for (int z = 0; z < zSegments; z++)
        {
            for (int x = 0; x < xSegments; x++)
            {
                // Pobieramy wcześniej wylosowane wysokości dla 4 rogów danego kwadratu
                float h_bl = heights[x, z];         // bottom-left
                float h_br = heights[x + 1, z];     // bottom-right
                float h_tl = heights[x, z + 1];     // top-left
                float h_tr = heights[x + 1, z + 1]; // top-right

                // Definiujemy narożniki z poprawnymi wysokościami
                Vector3 bl = new Vector3(startX + x * xStep, h_bl, startZ + z * zStep);
                Vector3 br = new Vector3(startX + (x + 1) * xStep, h_br, startZ + z * zStep);
                Vector3 tl = new Vector3(startX + x * xStep, h_tl, startZ + (z + 1) * zStep);
                Vector3 tr = new Vector3(startX + (x + 1) * xStep, h_tr, startZ + (z + 1) * zStep);

                // Trójkąt 1
                vertices[v] = bl; vertices[v + 1] = tl; vertices[v + 2] = br;
                // Trójkąt 2
                vertices[v + 3] = tl; vertices[v + 4] = tr; vertices[v + 5] = br;

                for (int i = 0; i < 6; i++) triangles[v + i] = v + i;
                v += 6;
            }
        }

        mesh.vertices = vertices;
        mesh.triangles = triangles;

        // KROK D: Przeliczenie danych niezbędnych dla fizyki i oświetlenia
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();

        return mesh;
    }
}
