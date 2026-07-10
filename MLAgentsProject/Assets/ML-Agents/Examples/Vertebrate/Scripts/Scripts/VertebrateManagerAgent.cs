using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgentsExamples;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;

//[RequireComponent(typeof(JointDriveController))] // Required to set joint forces
public class VertebrateManagerAgent : Agent
{

    [Header("Ostateczny Cel")]
    public Transform finalTargetPrefab;
    private Transform finalTarget;

    [Header("Referencja do Robotnika")]
    public VertebrateWorkerAgent worker;

    // TO JEST NASZ WIRTUALNY JOYSTICK (Przekazujemy to Robotnikowi)
    public Vector3 JoystickCommand { get; private set; }

    private float lastDistanceToTarget;

    public override void Initialize()
    {
        finalTarget = Instantiate(finalTargetPrefab, transform.position, Quaternion.identity, transform.parent);
    }

    public override void OnEpisodeBegin()
    {
        finalTarget.localPosition = new Vector3(Random.Range(-15f, 15f), 0, Random.Range(-15f, 15f));
        lastDistanceToTarget = Vector3.Distance(worker.body.position, finalTarget.position);

        JoystickCommand = Vector3.zero; // Reset pada
        worker.EndEpisode();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // 1. Obserwacja: Gdzie jest cel względem korpusu robota (Zmysł kierunku)
        Vector3 dirToTarget = (finalTarget.position - worker.body.position).normalized;
        sensor.AddObservation(dirToTarget.x);
        sensor.AddObservation(dirToTarget.z);

        // 2. Obserwacja: Jak daleko jest do celu
        sensor.AddObservation(Vector3.Distance(worker.body.position, finalTarget.position));
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // Menedżer wychyla gałkę pada (X i Z od -1 do 1)
        float joyX = actionBuffers.ContinuousActions[0];
        float joyZ = actionBuffers.ContinuousActions[1];

        // Zapisujemy komendę i ucinamy do długości 1 (żeby nie pękł silnik fizyczny)
        JoystickCommand = Vector3.ClampMagnitude(new Vector3(joyX, 0, joyZ), 1f);

        // --- NAGRODA MENEDŻERA ---
        float currentDistance = Vector3.Distance(worker.body.position, finalTarget.position);
        if (currentDistance < lastDistanceToTarget)
        {
            AddReward(0.1f); // Idziemy w stronę celu!
        }
        else
        {
            AddReward(-0.02f); // Błądzimy
        }
        lastDistanceToTarget = currentDistance;

        if (currentDistance < 2.0f)
        {
            AddReward(1.0f);
            EndEpisode();
        }
    }
}
