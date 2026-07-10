using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgentsExamples;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;

[RequireComponent(typeof(JointDriveController))]
public class VertebrateAgentNFExtendedLooking : Agent
{
    [Header("Walk Speed")]
    [Range(0.1f, m_maxWalkingSpeed)]
    [SerializeField]
    private float m_TargetWalkingSpeed = m_maxWalkingSpeed;
    const float m_maxWalkingSpeed = 15;

    public float TargetWalkingSpeed
    {
        get { return m_TargetWalkingSpeed; }
        set { m_TargetWalkingSpeed = Mathf.Clamp(value, .1f, m_maxWalkingSpeed); }
    }

    [Header("Target To Walk Towards")]
    public Transform TargetPrefab;
    private Transform m_Target;

    [Header("Body Parts")]
    [Space(10)]
    public Transform body;
    public Transform leg0Upper;
    public Transform leg0Lower;
    public Transform leg1Upper;
    public Transform leg1Lower;
    public Transform leg2Upper;
    public Transform leg2Lower;
    public Transform leg3Upper;
    public Transform leg3Lower;

    public Transform segment0; // Założenie: to jest "Głowa" agenta
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

    [Header("Leg Raycast Settings")]
    [Space(10)]
    public LayerMask groundLayer = ~0;
    public float legRaycastDistance = 3.0f;

    public TerrainWithMaterial terrainWithMaterial;

    [Header("Penalties")]
    [Space(10)]
    public float bellyTouchPenalty = -0.02f;
    public float energyPenaltyWeight = -0.000001f;
    public float maxBentDuration = 5.0f;
    public float bentAngleThreshold = 15.0f;
    public float overBentPenalty = -0.01f;
    public float turnNeededThreshold = 10.0f;
    public float notBendingWhenTurningPenalty = -0.01f;
    public float maxBellyTouchDuration = 2.0f;

    private void Start()
    {
        terrainWithMaterial.height = 1; //zmniejszone żeby agent przez kamere widział cel
    }

    public override void Initialize()
    {
        SpawnTarget(TargetPrefab, transform.position + new Vector3(0f, 2f, 0f));

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

    void SpawnTarget(Transform prefab, Vector3 pos)
    {
        m_Target = Instantiate(prefab, pos, Quaternion.identity, transform.parent);
    }

    public override void OnEpisodeBegin()
    {
        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            bodyPart.Reset(bodyPart);
        }

        body.rotation = Quaternion.Euler(0, Random.Range(0.0f, 360.0f), 0);

        UpdateOrientationObjects();

        TargetWalkingSpeed = Random.Range(7f, m_maxWalkingSpeed);
        terrainWithMaterial.generateRandomTerrain();
    }

    public void CollectObservationBodyPart(BodyPart bp, VectorSensor sensor)
    {
        sensor.AddObservation(bp.groundContact.touchingGround);

        if (bp.rb.transform != body)
        {
            sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit);
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // [ZMIANA] USUNIĘTO wszystkie obserwacje, które zdradzały pozycję i kierunek celu!
        // Agent musi teraz polegać wyłącznie na kamerze i swoim zmyśle równowagi.

        var avgVel = GetAvgVelocity();

        // Zostawiamy jedynie informację o tym, jak szybko i w jakim kierunku agent aktualnie się porusza 
        // (względem jego własnego "przodu")
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(avgVel));

        RaycastHit hit;
        float maxRaycastDist = 10;
        if (Physics.Raycast(body.position, Vector3.down, out hit, maxRaycastDist, groundLayer))
        {
            sensor.AddObservation(hit.distance / maxRaycastDist);
        }
        else
            sensor.AddObservation(1);

        CollectLegTerrainObservation(leg0Lower, sensor);
        CollectLegTerrainObservation(leg1Lower, sensor);
        CollectLegTerrainObservation(leg2Lower, sensor);
        CollectLegTerrainObservation(leg3Lower, sensor);

        foreach (var bodyPart in m_JdController.bodyPartsList)
        {
            CollectObservationBodyPart(bodyPart, sensor);
        }
    }

    private void CollectLegTerrainObservation(Transform legTransform, VectorSensor sensor)
    {
        RaycastHit hit;
        if (Physics.Raycast(legTransform.position, Vector3.down, out hit, legRaycastDistance, groundLayer))
        {
            sensor.AddObservation(hit.distance / legRaycastDistance);
            Vector3 localSurfaceNormal = m_OrientationCube.transform.InverseTransformDirection(hit.normal);
            sensor.AddObservation(localSurfaceNormal);
        }
        else
        {
            sensor.AddObservation(1f);
            sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(Vector3.up));
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
        bpDict[leg0Lower].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[leg1Lower].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[leg2Lower].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[leg3Lower].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);

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

    void FixedUpdate()
    {
        UpdateOrientationObjects();

        if (useFootGroundedVisualization)
        {
            foot0.material = m_JdController.bodyPartsDict[leg0Lower].groundContact.touchingGround ? groundedMaterial : unGroundedMaterial;
            foot1.material = m_JdController.bodyPartsDict[leg1Lower].groundContact.touchingGround ? groundedMaterial : unGroundedMaterial;
            foot2.material = m_JdController.bodyPartsDict[leg2Lower].groundContact.touchingGround ? groundedMaterial : unGroundedMaterial;
            foot3.material = m_JdController.bodyPartsDict[leg3Lower].groundContact.touchingGround ? groundedMaterial : unGroundedMaterial;
        }

        AddReward(0.001f);

        float uprightDot = Vector3.Dot(body.up, Vector3.up);

        if (uprightDot < 0.1f)
        {
            AddReward(-1.0f);
            EndEpisode();
            return;
        }

        // [ZMIANA] System nagród obliczany "od zera" na podstawie prawdziwej wektorowej drogi do celu.
        // Mimo że agent tego nie widzi w liczbach, system nagradza go, jeśli podąża w tę stronę na podstawie obrazu z kamery.
        Vector3 dirToTarget = (m_Target.position - body.position).normalized;
        dirToTarget.y = 0;
        if (dirToTarget == Vector3.zero) dirToTarget = body.forward;

        var matchSpeedReward = GetMatchingVelocityReward(dirToTarget * TargetWalkingSpeed, GetAvgVelocity());
        var lookAtTargetReward = Mathf.Pow(((Vector3.Dot(dirToTarget, segment0.forward) + 1) * .25F) + ((Vector3.Dot(dirToTarget, body.forward) + 1) * .25F), 2);

        float mainGoalReward = matchSpeedReward * lookAtTargetReward;
        AddReward(mainGoalReward);

        bool isBellyTouching = m_JdController.bodyPartsDict[body].groundContact.touchingGround ||
                               m_JdController.bodyPartsDict[segment0].groundContact.touchingGround ||
                               m_JdController.bodyPartsDict[segment1].groundContact.touchingGround ||
                               m_JdController.bodyPartsDict[segment2].groundContact.touchingGround;

        if (!isBellyTouching)
        {
            AddReward(0.05f * mainGoalReward);
        }

        // [ZMIANA] Ocena wygięcia kręgosłupa względem rzeczywistego kierunku do celu
        Transform[] spineSegments = { segment0, segment1, segment2 };
        float turnNeeded = Vector3.SignedAngle(body.forward, dirToTarget, Vector3.up);
        float signNeeded = Mathf.Sign(turnNeeded);
        bool isTurning = Mathf.Abs(turnNeeded) > turnNeededThreshold;

        float spineAlignmentReward = 0f;

        foreach (var seg in spineSegments)
        {
            float segAngle = Vector3.SignedAngle(body.forward, seg.forward, Vector3.up);

            if (isTurning)
            {
                if (Mathf.Sign(segAngle) == signNeeded && Mathf.Abs(segAngle) >= 1.5f && Mathf.Abs(segAngle) < bentAngleThreshold)
                {
                    spineAlignmentReward += 0.02f;
                }
            }
            else
            {
                if (Mathf.Abs(segAngle) < 5.0f)
                {
                    spineAlignmentReward += 0.03f;
                }
            }
        }
        AddReward(spineAlignmentReward);

        float energyUsed = 0f;
        foreach (var bp in m_JdController.bodyPartsList)
        {
            if (bp.rb.transform != body)
            {
                float normalizedStrength = bp.currentStrength / m_JdController.maxJointForceLimit;
                energyUsed += normalizedStrength * normalizedStrength;
            }
        }
        AddReward(energyPenaltyWeight * energyUsed);
    }

    void UpdateOrientationObjects()
    {
        // [ZMIANA] OrientationCube stabilizuje teraz orientację własnego ciała agenta, a nie kierunek na cel.
        // Dzięki temu lokalne obserwacje (raycasty, prędkość) nie będą zdradzać kierunku do celu.
        Vector3 forwardPlane = body.forward;
        forwardPlane.y = 0;
        if (forwardPlane.sqrMagnitude > 0)
        {
            m_OrientationCube.transform.rotation = Quaternion.LookRotation(forwardPlane, Vector3.up);
        }
        m_OrientationCube.transform.position = body.position;

        if (m_DirectionIndicator)
        {
            // Opcjonalnie: Strzałka nadal wizualnie pokazuje cel na ziemi
            Vector3 dirToTarget = m_Target.position - body.position;
            dirToTarget.y = 0;
            if (dirToTarget.sqrMagnitude > 0)
            {
                m_DirectionIndicator.transform.rotation = Quaternion.LookRotation(dirToTarget, Vector3.up);
            }
            m_DirectionIndicator.transform.position = body.position;
        }
    }

    Vector3 GetAvgVelocity()
    {
        Vector3 vel = m_JdController.bodyPartsDict[body].rb.linearVelocity;
        vel.y = 0;
        return vel;
    }

    public float GetMatchingVelocityReward(Vector3 velocityGoal, Vector3 actualVelocity)
    {
        var velDeltaMagnitude = Mathf.Clamp(Vector3.Distance(actualVelocity, velocityGoal), 0, TargetWalkingSpeed);
        return Mathf.Pow(1 - Mathf.Pow(velDeltaMagnitude / TargetWalkingSpeed, 2), 2);
    }

    public void TouchedTarget()
    {
        AddReward(1f);
    }
}
