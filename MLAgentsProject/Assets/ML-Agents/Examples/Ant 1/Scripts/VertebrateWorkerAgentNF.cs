using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgentsExamples;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;

[RequireComponent(typeof(JointDriveController))] // Required to set joint forces
public class VertebrateWorkerAgentNF : Agent
{
    // --- NOWE: Połączenie z Menedżerem ---
    [Header("Referencja do Menedżera")]
    public VertebrateManagerAgent manager;

    [Header("Walk Speed")]
    [Range(0.1f, m_maxWalkingSpeed)]
    public float TargetWalkingSpeed = 10f;
    const float m_maxWalkingSpeed = 15;

    [Header("Body Parts")]
    [Space(10)] public Transform body;
    public Transform leg0Upper;
    public Transform leg0Lower;
    public Transform leg1Upper;
    public Transform leg1Lower;
    public Transform leg2Upper;
    public Transform leg2Lower;
    public Transform leg3Upper;
    public Transform leg3Lower;
    public Transform segment0;
    public Transform segment1;
    public Transform segment2;

    OrientationCubeController m_OrientationCube;
    DirectionIndicator m_DirectionIndicator;
    JointDriveController m_JdController;

    [Header("Foot Grounded Visualization")]
    [Space(10)]
    public bool useFootGroundedVisualization;
    public MeshRenderer foot0;
    public MeshRenderer foot1;
    public MeshRenderer foot2;
    public MeshRenderer foot3;
    public Material groundedMaterial;
    public Material unGroundedMaterial;

    public override void Initialize()
    {
        m_OrientationCube = GetComponentInChildren<OrientationCubeController>();
        m_DirectionIndicator = GetComponentInChildren<DirectionIndicator>();
        m_JdController = GetComponent<JointDriveController>();

        m_JdController.SetupBodyPart(body);
        m_JdController.SetupBodyPart(leg0Upper);
        m_JdController.SetupBodyPart(leg0Lower);
        m_JdController.SetupBodyPart(leg1Upper);
        m_JdController.SetupBodyPart(leg1Lower);
        m_JdController.SetupBodyPart(leg2Upper);
        m_JdController.SetupBodyPart(leg2Lower);
        m_JdController.SetupBodyPart(leg3Upper);
        m_JdController.SetupBodyPart(leg3Lower);
        m_JdController.SetupBodyPart(segment0);
        m_JdController.SetupBodyPart(segment1);
        m_JdController.SetupBodyPart(segment2);
    }

    public override void OnEpisodeBegin()
    {
        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            bodyPart.Reset(bodyPart);
        }

        body.rotation = Quaternion.Euler(0, Random.Range(0.0f, 360.0f), 0);
        UpdateOrientationObjects();
        TargetWalkingSpeed = 10;
    }

    public void CollectObservationBodyPart(BodyPart bp, VectorSensor sensor)
    {
        sensor.AddObservation(bp.groundContact.touchingGround);
        if (bp.rb.transform != body)
        {
            sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit);
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var bpDict = m_JdController.bodyPartsDict;
        var continuousActions = actionBuffers.ContinuousActions;
        var i = -1;

        bpDict[leg0Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[leg1Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[leg2Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[leg3Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[leg0Lower].SetJointTargetRotation(continuousActions[++i], 0, 0);
        bpDict[leg1Lower].SetJointTargetRotation(continuousActions[++i], 0, 0);
        bpDict[leg2Lower].SetJointTargetRotation(continuousActions[++i], 0, 0);
        bpDict[leg3Lower].SetJointTargetRotation(continuousActions[++i], 0, 0);

        bpDict[segment0].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[segment1].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[segment2].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);

        bpDict[leg0Upper].SetJointStrength(continuousActions[++i]);
        bpDict[leg1Upper].SetJointStrength(continuousActions[++i]);
        bpDict[leg2Upper].SetJointStrength(continuousActions[++i]);
        bpDict[leg3Upper].SetJointStrength(continuousActions[++i]);
        bpDict[leg0Lower].SetJointStrength(continuousActions[++i]);
        bpDict[leg1Lower].SetJointStrength(continuousActions[++i]);
        bpDict[leg2Lower].SetJointStrength(continuousActions[++i]);
        bpDict[leg3Lower].SetJointStrength(continuousActions[++i]);

        bpDict[segment0].SetJointStrength(continuousActions[++i]);
        bpDict[segment1].SetJointStrength(continuousActions[++i]);
        bpDict[segment2].SetJointStrength(continuousActions[++i]);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        var avgVel = GetAvgVelocity();

        // Obliczamy docelowy wektor prędkości (Wychylenie Joysticka * Max Prędkość)
        Vector3 targetVelocityVector = manager.JoystickCommand * TargetWalkingSpeed;

        // 1. Zmysł: Różnica między tym jak szybko idziemy, a jak szybko każe Menedżer
        sensor.AddObservation(Vector3.Distance(targetVelocityVector, avgVel));

        // 2. Zmysł: Aktualna prędkość lokalna
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(avgVel));

        // 3. Zmysł: Docelowa prędkość w układzie lokalnym (żeby robot wiedział w którą stronę skręcić)
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(targetVelocityVector));

        // 4. Zmysł: Aktualne wychylenie gałki (surowe)
        sensor.AddObservation(manager.JoystickCommand.x);
        sensor.AddObservation(manager.JoystickCommand.z);

        // Zabezpieczenie przed upadkiem (promień w dół)
        RaycastHit hit;
        float maxRaycastDist = 10;
        if (Physics.Raycast(body.position, Vector3.down, out hit, maxRaycastDist))
            sensor.AddObservation(hit.distance / maxRaycastDist);
        else
            sensor.AddObservation(1);

        foreach (var bodyPart in m_JdController.bodyPartsList)
        {
            CollectObservationBodyPart(bodyPart, sensor);
        }
    }

    void FixedUpdate()
    {
        UpdateOrientationObjects();
        //CheckIfFell();

        // ... [Kolorowanie stóp zostaje bez zmian] ...

        var avgVel = GetAvgVelocity();

        // Wektor, który Menedżer CHCE żebyśmy osiągnęli
        Vector3 targetVelocityVector = manager.JoystickCommand * TargetWalkingSpeed;

        // NAGRODA: Im mniejsza różnica między prędkością fizyczną a wektorem joysticka, tym bliżej 1.0 nagrody.
        float velDelta = Vector3.Distance(avgVel, targetVelocityVector);
        float matchReward = Mathf.Pow(1f - Mathf.Clamp01(velDelta / TargetWalkingSpeed), 2f);

        // Robotnik dostaje punkty TYLKO za słuchanie się poleceń Joysticka
        AddReward(matchReward * 0.1f);
    }

    void UpdateOrientationObjects()
    {
        // BARDZO WAŻNE DLA STABILNOŚCI SIECI!
        // Kostka orientacji "patrzy" w kierunku, w którym wychylony jest Joystick.
        // Dzięki temu sieć Robotnika zawsze uczy się, że "przód" to tam, gdzie każe iść Menedżer.
        if (manager.JoystickCommand.sqrMagnitude > 0.1f)
        {
            // Tymczasowo tworzymy wirtualny punkt w stronę, w którą każe iść Menedżer
            Vector3 lookDir = manager.JoystickCommand.normalized;
            m_OrientationCube.transform.rotation = Quaternion.LookRotation(lookDir, Vector3.up);
        }

        if (m_DirectionIndicator)
        {
            m_DirectionIndicator.MatchOrientation(m_OrientationCube.transform);
        }
    }

    Vector3 GetAvgVelocity()
    {
        Vector3 velSum = Vector3.zero;
        int numOfRb = 0;
        foreach (var item in m_JdController.bodyPartsList)
        {
            numOfRb++;
            velSum += item.rb.linearVelocity;
        }
        return velSum / numOfRb;
    }

    public float GetMatchingVelocityReward(Vector3 velocityGoal, Vector3 actualVelocity)
    {
        var velDeltaMagnitude = Mathf.Clamp(Vector3.Distance(actualVelocity, velocityGoal), 0, TargetWalkingSpeed);
        return Mathf.Pow(1 - Mathf.Pow(velDeltaMagnitude / TargetWalkingSpeed, 2), 2);
    }

    //void CheckIfFell()
    //{
    //    // Przykładowy test upadku: jeśli ciało zbytnio obniży się do ziemi
    //    if (body.transform.localPosition.y < 0.5f)
    //    {
    //        AddReward(-1f);

    //        // Komunikacja z Menedżerem: "Szefie, upadłem, musimy zacząć od nowa"
    //        manager.AddReward(-0.5f);
    //        manager.EndEpisode();

    //        EndEpisode();
    //    }
    //}
}
