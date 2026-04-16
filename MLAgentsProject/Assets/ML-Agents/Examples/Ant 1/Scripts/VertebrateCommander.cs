using UnityEngine;

public class VertebateCommander : MonoBehaviour
{
    //[Header("Ostateczny Cel")]
    //public Transform finalTargetPrefab;
    //private Transform finalTarget;

    //[Header("Referencja do Robotnika")]
    //public VertebateHybridAgent worker;

    //public Vector3 JoystickCommand { get; private set; }

    //void Start()
    //{
    //    finalTarget = Instantiate(finalTargetPrefab, transform.position, Quaternion.identity, transform.parent);
    //    ResetTarget();
    //}

    //public void ResetTarget()
    //{
    //    if (finalTarget != null && worker != null)
    //    {
    //        finalTarget.localPosition = new Vector3(Random.Range(-15f, 15f), 0, Random.Range(-15f, 15f));
    //        JoystickCommand = Vector3.zero;
    //    }
    //}

    //void FixedUpdate()
    //{
    //    if (worker == null || finalTarget == null) return;

    //    // wyliczamy wektor od robota do celu
    //    Vector3 dirToTarget = (finalTarget.position - worker.body.position);
    //    float distance = dirToTarget.magnitude;

    //    // Jeśli robot dotknął celu (jest wystarczająco blisko)
    //    if (distance < 2.0f)
    //    {
    //        worker.ReachedTarget();
    //    }
    //    else
    //    {
    //        JoystickCommand = new Vector3(dirToTarget.x, 0, dirToTarget.z).normalized;
    //    }
    //}

    //void OnDrawGizmos()
    //{
    //    if (worker != null && finalTarget != null)
    //    {
    //        Gizmos.color = Color.green;
    //        Gizmos.DrawLine(worker.body.position, finalTarget.position);
    //    }
    //}
}
