using UnityEngine;

public class Test : MonoBehaviour
{
    public ConfigurableJoint j;
    void Update()
    {
        var rot = Quaternion.Euler(Mathf.Sin(Time.time) * 30, 0, 0);
        j.targetRotation = rot;
    }
}
