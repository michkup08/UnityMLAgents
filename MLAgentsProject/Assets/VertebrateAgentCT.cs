using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgentsExamples;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;

[RequireComponent(typeof(JointDriveController))]
public class VertebrateAgentCT : Agent
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

    [Header("Body Parts")][Space(10)] public Transform body;
    public Transform leg0Upper;
    public Transform leg0Lower;
    public Transform leg0Last;
    public Transform leg1Upper;
    public Transform leg1Lower;
    public Transform leg1Last;
    public Transform leg2Upper;
    public Transform leg2Lower;
    public Transform leg2Last;
    public Transform leg3Upper;
    public Transform leg3Lower;
    public Transform leg3Last;

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

    [Header("Leg Raycast Settings")]
    [Space(10)]
    [Tooltip("Warstwa, na której znajduje się ziemia. Ustaw to tak, aby raycast nie uderzał w same nogi agenta!")]
    public LayerMask groundLayer = ~0;
    public float legRaycastDistance = 3.0f;

    public TerrainWithMaterial terrainWithMaterial;

    [Header("Penalties")]
    [Space(10)]
    [Tooltip("Kara naliczana co krok, gdy brzuch/kręgosłup dotyka ziemi")]
    public float bellyTouchPenalty = -0.02f;

    [Tooltip("Jak długo agent może być zgięty (w sekundach) zanim zacznie otrzymywać karę")]
    public float maxBentDuration = 5.0f;
    [Tooltip("Kąt w stopniach uznawany za 'zgięcie' kręgosłupa (odchylenie segmentu 2 od głównego ciała)")]
    public float bentAngleThreshold = 15.0f;
    [Tooltip("Kara naliczana co krok, gdy agent pozostaje zgięty za długo")]
    public float overBentPenalty = -0.01f;

    [Tooltip("Kąt (w stopniach) błędu w kierunku do celu, od którego oczekujemy zgięcia kręgosłupa")]
    public float turnNeededThreshold = 10.0f;
    [Tooltip("Kara za brak odpowiedniego zgięcia (bocznego) kręgosłupa podczas skręcania")]
    public float notBendingWhenTurningPenalty = -0.01f;

    public float maxBellyTouchDuration = 2.0f;

    public bool threePartLegs = false;

    private float[] recentFootContacts = new float[4];

    public override void Initialize()
    {
        terrainWithMaterial.height = 0;
        terrainWithMaterial.yOffset = 1;

        SpawnTarget(TargetPrefab, transform.position + new Vector3(0f, 4f, 0f));

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

        if (threePartLegs)
        {
            m_JdController.SetupBodyPart(leg0Last);
            m_JdController.SetupBodyPart(leg1Last);
            m_JdController.SetupBodyPart(leg2Last);
            m_JdController.SetupBodyPart(leg3Last);
        }

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

        for (int i = 0; i < recentFootContacts.Length; i++)
        {
            recentFootContacts[i] = 0f;
        }

        terrainWithMaterial.height = 0;
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
        var cubeForward = m_OrientationCube.transform.forward;

        var velGoal = cubeForward * TargetWalkingSpeed;
        var avgVel = GetAvgVelocity();

        sensor.AddObservation(Vector3.Distance(velGoal, avgVel));
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(avgVel));
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(velGoal));
        sensor.AddObservation(Quaternion.FromToRotation(body.forward, cubeForward));

        sensor.AddObservation(m_OrientationCube.transform.InverseTransformPoint(m_Target.transform.position));

        RaycastHit hit;
        float maxRaycastDist = 10;
        if (Physics.Raycast(body.position, Vector3.down, out hit, maxRaycastDist, groundLayer))
        {
            sensor.AddObservation(hit.distance / maxRaycastDist);
        }
        else
            sensor.AddObservation(1);

        foreach (var bodyPart in m_JdController.bodyPartsList)
        {
            CollectObservationBodyPart(bodyPart, sensor);
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

        if (threePartLegs)
        {
            bpDict[leg0Last].SetJointTargetRotation(continuousActions[++i], 0, 0);
            bpDict[leg1Last].SetJointTargetRotation(continuousActions[++i], 0, 0);
            bpDict[leg2Last].SetJointTargetRotation(continuousActions[++i], 0, 0);
            bpDict[leg3Last].SetJointTargetRotation(continuousActions[++i], 0, 0);
        }

        // Po nogach:
        bpDict[segment0].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[segment1].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[segment2].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);


        // Update joint strength
        bpDict[leg0Upper].SetJointStrength(continuousActions[++i]);
        bpDict[leg1Upper].SetJointStrength(continuousActions[++i]);
        bpDict[leg2Upper].SetJointStrength(continuousActions[++i]);
        bpDict[leg3Upper].SetJointStrength(continuousActions[++i]);
        bpDict[leg0Lower].SetJointStrength(continuousActions[++i]);
        bpDict[leg1Lower].SetJointStrength(continuousActions[++i]);
        bpDict[leg2Lower].SetJointStrength(continuousActions[++i]);
        bpDict[leg3Lower].SetJointStrength(continuousActions[++i]);

        if (threePartLegs)
        {
            bpDict[leg0Last].SetJointStrength(continuousActions[++i]);
            bpDict[leg1Last].SetJointStrength(continuousActions[++i]);
            bpDict[leg2Last].SetJointStrength(continuousActions[++i]);
            bpDict[leg3Last].SetJointStrength(continuousActions[++i]);
        }

        // Strength:
        bpDict[segment0].SetJointStrength(continuousActions[++i]);
        bpDict[segment1].SetJointStrength(continuousActions[++i]);
        bpDict[segment2].SetJointStrength(continuousActions[++i]);

    }

    void FixedUpdate()
    {
        UpdateOrientationObjects();

        if (useFootGroundedVisualization)
        {
            foot0.material = m_JdController.bodyPartsDict[leg0Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
            foot1.material = m_JdController.bodyPartsDict[leg1Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
            foot2.material = m_JdController.bodyPartsDict[leg2Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
            foot3.material = m_JdController.bodyPartsDict[leg3Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
        }

        AddReward(0.001f);

        float uprightDot = Vector3.Dot(body.up, Vector3.up);



        if (uprightDot < 0.1f)
        {
            AddReward(-1.0f);
            EndEpisode();
            return;
        }

        var cubeForward = m_OrientationCube.transform.forward;
        var matchSpeedReward = GetMatchingVelocityReward(cubeForward * TargetWalkingSpeed, GetAvgVelocity());
        var lookAtTargetReward = Mathf.Pow(((Vector3.Dot(cubeForward, segment0.forward) + 1) * .25F) + ((Vector3.Dot(cubeForward, body.forward) + 1) * .25F), 2);


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

        Transform[] spineSegments = { segment0, segment1, segment2 };
        float turnNeeded = Vector3.SignedAngle(body.forward, cubeForward, Vector3.up);
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
                    spineAlignmentReward += 0.1f;
                }
            }
            else
            {
                if (Mathf.Abs(segAngle) < 5.0f)
                {
                    spineAlignmentReward += 0.15f;
                }
            }
        }

        AddReward(spineAlignmentReward * mainGoalReward);

        Transform[] upperLegSegments = { leg0Upper, leg1Upper, leg2Upper, leg3Upper };
        if (threePartLegs)
        {
            upperLegSegments = new Transform[] { leg0Upper, leg1Upper, leg2Upper, leg3Upper, leg0Lower, leg1Lower, leg2Lower, leg3Lower };
        }
        foreach (var seg in upperLegSegments)
        {
            if (!m_JdController.bodyPartsDict[seg].groundContact.touchingGround)
            {
                AddReward(0.2f * mainGoalReward);
            }
        }

        Transform[] allBodySegments = { body, segment0, segment1, segment2 };
        float flatPostureReward = 0f;

        foreach (var seg in allBodySegments)
        {
            float dotUp = Vector3.Dot(seg.up, Vector3.up);

            if (dotUp > 0.5f)
            {
                float flatnessMultiplier = Mathf.InverseLerp(0.5f, 1.0f, dotUp);

                flatPostureReward += flatnessMultiplier * 0.1f;
            }
        }

        AddReward(flatPostureReward * mainGoalReward);

        Transform[] feetSegments = { leg0Lower, leg1Lower, leg2Lower, leg3Lower };

        if (threePartLegs)
        {
            feetSegments = new Transform[] { leg0Last, leg1Last, leg2Last, leg3Last };
        }

        
        float emaAlpha = 0.02f;

        for (int i = 0; i < feetSegments.Length; i++)
        {
            float isTouching = m_JdController.bodyPartsDict[feetSegments[i]].groundContact.touchingGround ? 1f : 0f;

            recentFootContacts[i] = Mathf.Lerp(recentFootContacts[i], isTouching, emaAlpha);
        }

        float maxContact = Mathf.Max(recentFootContacts[0], recentFootContacts[1], recentFootContacts[2], recentFootContacts[3]);
        float minContact = Mathf.Min(recentFootContacts[0], recentFootContacts[1], recentFootContacts[2], recentFootContacts[3]);

        float contactDifference = maxContact - minContact;

        float symmetryReward = 1f - contactDifference;

        
        AddReward(symmetryReward * 0.1f * mainGoalReward);

    }

    void UpdateOrientationObjects()
    {
        m_OrientationCube.UpdateOrientation(segment0, m_Target);
        if (m_DirectionIndicator)
        {
            m_DirectionIndicator.MatchOrientation(m_OrientationCube.transform);
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

