using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgentsExamples;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;

[RequireComponent(typeof(JointDriveController))] // Required to set joint forces
public class VertebrateAgentNFExtended : Agent
{

    [Header("Walk Speed")]
    [Range(0.1f, m_maxWalkingSpeed)]
    [SerializeField]
    [Tooltip(
        "The speed the agent will try to match.\n\n" +
        "TRAINING:\n" +
        "For VariableSpeed envs, this value will randomize at the start of each training episode.\n" +
        "Otherwise the agent will try to match the speed set here.\n\n" +
        "INFERENCE:\n" +
        "During inference, VariableSpeed agents will modify their behavior based on this value " +
        "whereas the CrawlerDynamic & CrawlerStatic agents will run at the speed specified during training "
    )]
    //The walking speed to try and achieve
    private float m_TargetWalkingSpeed = m_maxWalkingSpeed;

    const float m_maxWalkingSpeed = 15; //The max walking speed

    //The current target walking speed. Clamped because a value of zero will cause NaNs
    public float TargetWalkingSpeed
    {
        get { return m_TargetWalkingSpeed; }
        set { m_TargetWalkingSpeed = Mathf.Clamp(value, .1f, m_maxWalkingSpeed); }
    }

    //The direction an agent will walk during training.
    [Header("Target To Walk Towards")]
    public Transform TargetPrefab; //Target prefab to use in Dynamic envs
    private Transform m_Target; //Target the agent will walk towards during training.

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


    //This will be used as a stabilized model space reference point for observations
    //Because ragdolls can move erratically during training, using a stabilized reference transform improves learning
    OrientationCubeController m_OrientationCube;

    //The indicator graphic gameobject that points towards the target
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

    // [NOWE] - Dodano ustawienia dla Raycastów nóg
    [Header("Leg Raycast Settings")]
    [Space(10)]
    [Tooltip("Warstwa, na której znajduje się ziemia. Ustaw to tak, aby raycast nie uderzał w same nogi agenta!")]
    public LayerMask groundLayer = ~0;
    public float legRaycastDistance = 3.0f; // Jak daleko w dół mają "patrzeć" nogi

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

    // Zmienna śledząca czas przebywania w zgięciu
    //private float m_BentTimer = 0f;

    //private float m_BellyTouchTimer = 0f;
    public float maxBellyTouchDuration = 2.0f;

    public bool threePartLegs = false;

    public override void Initialize()
    {
        SpawnTarget(TargetPrefab, transform.position + new Vector3(0f, 4f, 0f)); //spawn target

        m_OrientationCube = GetComponentInChildren<OrientationCubeController>();
        m_DirectionIndicator = GetComponentInChildren<DirectionIndicator>();
        m_JdController = GetComponent<JointDriveController>();

        //Setup each body part
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

    /// <summary>
    /// Spawns a target prefab at pos
    /// </summary>
    /// <param name="prefab"></param>
    /// <param name="pos"></param>
    void SpawnTarget(Transform prefab, Vector3 pos)
    {
        m_Target = Instantiate(prefab, pos, Quaternion.identity, transform.parent);
    }

    /// <summary>
    /// Loop over body parts and reset them to initial conditions.
    /// </summary>
    public override void OnEpisodeBegin()
    {
        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            bodyPart.Reset(bodyPart);
        }

        //Random start rotation to help generalize
        body.rotation = Quaternion.Euler(0, Random.Range(0.0f, 360.0f), 0);

        UpdateOrientationObjects();

        //Set our goal walking speed
        TargetWalkingSpeed = Random.Range(7f, m_maxWalkingSpeed);

        terrainWithMaterial.generateRandomTerrain();
    }

    /// <summary>
    /// Add relevant information on each body part to observations.
    /// </summary>
    public void CollectObservationBodyPart(BodyPart bp, VectorSensor sensor)
    {
        //GROUND CHECK
        sensor.AddObservation(bp.groundContact.touchingGround); // Is this bp touching the ground

        if (bp.rb.transform != body)
        {
            sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit);
        }
    }

    /// <summary>
    /// Loop over body parts to add them to observation.
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
        var cubeForward = m_OrientationCube.transform.forward;

        //velocity we want to match
        var velGoal = cubeForward * TargetWalkingSpeed;
        //ragdoll's avg vel
        var avgVel = GetAvgVelocity();

        //current ragdoll velocity. normalized
        sensor.AddObservation(Vector3.Distance(velGoal, avgVel));
        //avg body vel relative to cube
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(avgVel));
        //vel goal relative to cube
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(velGoal));
        //rotation delta
        //sensor.AddObservation(Quaternion.FromToRotation(body.forward, cubeForward));
        sensor.AddObservation(Quaternion.FromToRotation(body.forward, cubeForward));

        //Add pos of target relative to orientation cube
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformPoint(m_Target.transform.position));

        // Dodano maskę warstw, żeby raycast środka ciała również ignorował własne collidery
        RaycastHit hit;
        float maxRaycastDist = 10;
        if (Physics.Raycast(body.position, Vector3.down, out hit, maxRaycastDist, groundLayer))
        {
            sensor.AddObservation(hit.distance / maxRaycastDist);
        }
        else
            sensor.AddObservation(1);

        // Zbieranie danych o terenie z każdej nogi (dolne segmenty nóg)
        CollectLegTerrainObservation(leg0Lower, sensor);
        CollectLegTerrainObservation(leg1Lower, sensor);
        CollectLegTerrainObservation(leg2Lower, sensor);
        CollectLegTerrainObservation(leg3Lower, sensor);

        foreach (var bodyPart in m_JdController.bodyPartsList)
        {
            CollectObservationBodyPart(bodyPart, sensor);
        }
    }

    // [NOWE] Metoda do strzelania raycastem i zbierania obserwacji dla pojedynczej kończyny
    private void CollectLegTerrainObservation(Transform legTransform, VectorSensor sensor)
    {
        RaycastHit hit;
        if (Physics.Raycast(legTransform.position, Vector3.down, out hit, legRaycastDistance, groundLayer))
        {
            // 1. Znormalizowana odległość do podłoża (1 obserwacja)
            sensor.AddObservation(hit.distance / legRaycastDistance);

            // 2. Kąt / Normalna podłoża (3 obserwacje). 
            // Konwertujemy wektor normalny do przestrzeni OrientationCube, dzięki czemu
            // agent uczy się kierunków "względem tego, w którą stronę idzie", a nie absolutnych współrzędnych świata.
            Vector3 localSurfaceNormal = m_OrientationCube.transform.InverseTransformDirection(hit.normal);
            sensor.AddObservation(localSurfaceNormal);
        }
        else
        {
            // Jeśli raycast niczego nie dotknie, zakładamy że ziemia jest max daleko i jest płaska
            sensor.AddObservation(1f);
            sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(Vector3.up));
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // The dictionary with all the body parts in it are in the jdController
        var bpDict = m_JdController.bodyPartsDict;

        var continuousActions = actionBuffers.ContinuousActions;
        var i = -1;
        // Pick a new target joint rotation
        bpDict[leg0Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[leg1Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[leg2Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[leg3Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[leg0Lower].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[leg1Lower].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[leg2Lower].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[leg3Lower].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);

        if(threePartLegs)
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

        // If enabled the feet will light up green when the foot is grounded.
        // This is just a visualization and isn't necessary for function
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

        //AddReward(matchSpeedReward * 0.1f);

        //AddReward(lookAtTargetReward * 0.1f);

        // --- POCZĄTEK FIXED UPDATE ---

        // 1. Sprawdzenie stanu krytycznego (Upadek)
        float uprightDot = Vector3.Dot(body.up, Vector3.up);

        

        if (uprightDot < 0.1f)
        {
            // Fatalny błąd - kończymy natychmiast
            AddReward(-1.0f);
            EndEpisode();
            return;
        }

        // 2. Nagroda główna (Prędkość i podążanie za celem)
        var cubeForward = m_OrientationCube.transform.forward;
        var matchSpeedReward = GetMatchingVelocityReward(cubeForward * TargetWalkingSpeed, GetAvgVelocity());
        // var lookAtTargetReward = Mathf.Pow((Vector3.Dot(cubeForward, body.forward) + 1) * .5F, 2); // do kwadratu zeby był bardziej restrykcyjny w kierunku
        var lookAtTargetReward = Mathf.Pow(((Vector3.Dot(cubeForward, segment0.forward) + 1) * .25F) + ((Vector3.Dot(cubeForward, body.forward) + 1) * .25F), 2);


        // Mnożymy przez główną wagę celów (np. 1.0, u Ciebie wcześniej było 25, co jest bardzo dużą wartością)
        float mainGoalReward = matchSpeedReward * lookAtTargetReward;
        AddReward(mainGoalReward);

        // 3. Nagroda za postawę (Belly Up) zamiast kary
        bool isBellyTouching = m_JdController.bodyPartsDict[body].groundContact.touchingGround ||
                               m_JdController.bodyPartsDict[segment0].groundContact.touchingGround ||
                               m_JdController.bodyPartsDict[segment1].groundContact.touchingGround ||
                               m_JdController.bodyPartsDict[segment2].groundContact.touchingGround;

        if (!isBellyTouching)
        {
            AddReward(0.05f * mainGoalReward);
        }

        // 4. Nagroda za prawidłowe zgięcie kręgosłupa
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
                // Skręcamy: Nagradzamy, jeśli segment wygina się w stronę skrętu
                if (Mathf.Sign(segAngle) == signNeeded && Mathf.Abs(segAngle) >= 1.5f && Mathf.Abs(segAngle) < bentAngleThreshold)
                {
                    spineAlignmentReward += 0.1f; // Mała nagroda za ładny łuk
                }
            }
            else
            {
                // Idziemy prosto: Nagradzamy, jeśli segment jest w miarę wyprostowany
                if (Mathf.Abs(segAngle) < 5.0f)
                {
                    spineAlignmentReward += 0.15f; // Mała nagroda za prosty kręgosłup
                }
            }
        }

        // Ponownie zabezpieczamy się przed "farmieniem" - nagroda za kręgosłup ma sens tylko podczas ruchu
        AddReward(spineAlignmentReward * mainGoalReward);

        Transform[] upperLegSegments = { leg0Upper, leg1Upper, leg2Upper, leg3Upper };
        if(threePartLegs)
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
            // dotUp wynosi 1.0 gdy segment leży idealnie płasko na ziemi
            float dotUp = Vector3.Dot(seg.up, Vector3.up);

            // Zakładamy próg tolerancji (np. 0.85 to lekki przechył). 
            // Nagradzamy tylko, jeśli segment jest w miarę poziomo.
            if (dotUp > 0.5f)
            {
                // Skalujemy wynik: dla dotUp = 0.5 daje 0, dla dotUp = 1.0 daje 1.0
                float flatnessMultiplier = Mathf.InverseLerp(0.5f, 1.0f, dotUp);

                // Dodajemy małą cząstkową nagrodę za ten konkretny segment
                flatPostureReward += flatnessMultiplier * 0.1f;
            }
        }

        // Przyznajemy nagrodę tylko w trakcie poprawnego realizowania głównego celu (ruchu)
        AddReward(flatPostureReward * mainGoalReward);


        ///////////////////////nizej jest system kar, poobujemy go zastąpić systemem nagród

        // 1. Kara za dotykanie brzuchem (lub segmentami kręgosłupa) ziemi
        //if (m_JdController.bodyPartsDict[body].groundContact.touchingGround ||
        //    m_JdController.bodyPartsDict[segment0].groundContact.touchingGround ||
        //    m_JdController.bodyPartsDict[segment1].groundContact.touchingGround ||
        //    m_JdController.bodyPartsDict[segment2].groundContact.touchingGround)
        //{
        //    AddReward(bellyTouchPenalty);
        //}

        //// 2. Kara za użycie energii (suma kwadratów znormalizowanej siły wszystkich stawów)
        //float energyUsed = 0f;
        //foreach (var bp in m_JdController.bodyPartsList)
        //{
        //    if (bp.rb.transform != body) // Główne ciało nie ma napędu
        //    {
        //        float normalizedStrength = bp.currentStrength / m_JdController.maxJointForceLimit;
        //        energyUsed += normalizedStrength * normalizedStrength;
        //    }
        //}
        //AddReward(energyPenaltyWeight * energyUsed);

        //// --- Analiza zgięcia kręgosłupa na płaszczyźnie poziomej (do kar 3 i 4) ---
        //// Używamy Vector3.SignedAngle wzdłuż osi Y, aby określić czy zgięcie jest w lewo/prawo i jak silne.
        //Transform[] spineSegments = { segment0, segment1, segment2 };
        //bool isOverBent = false;

        //// 3. Sprawdzamy, czy KTÓRYKOLWIEK segment jest wygięty za bardzo
        //foreach (var seg in spineSegments)
        //{
        //    float segAngle = Vector3.SignedAngle(body.forward, seg.forward, Vector3.up);
        //    if (Mathf.Abs(segAngle) > bentAngleThreshold)
        //    {
        //        isOverBent = true;
        //        break; // Wystarczy, że jeden jest wygięty za mocno
        //    }
        //}

        //// Obsługa timera dla kary 3
        //if (isOverBent)
        //{
        //    m_BentTimer += Time.fixedDeltaTime;
        //    if (m_BentTimer > maxBentDuration)
        //    {
        //        AddReward(overBentPenalty);
        //    }
        //}
        //else
        //{
        //    // Resetujemy timer, jeśli agent wyrównał wszystkie segmenty
        //    m_BentTimer = 0f;
        //}

        //// 4. Kara za niezgięcie kręgosłupa w odpowiednią stronę przy skręcaniu
        //float turnNeeded = Vector3.SignedAngle(body.forward, cubeForward, Vector3.up);

        //if (Mathf.Abs(turnNeeded) > turnNeededThreshold)
        //{
        //    float signNeeded = Mathf.Sign(turnNeeded);

        //    // Zamiast karać raz za całość, rozkładamy karę proporcjonalnie na każdy segment,
        //    // który "nie współpracuje" ze skrętem. Daje to łagodniejszy gradient uczenia.
        //    float penaltyPerSegment = notBendingWhenTurningPenalty / spineSegments.Length;

        //    foreach (var seg in spineSegments)
        //    {
        //        float segAngle = Vector3.SignedAngle(body.forward, seg.forward, Vector3.up);

        //        // Zmniejszyłem wymóg z 5f do 1.5f, ponieważ teraz wymagamy zgięcia od KAŻDEGO z 3 segmentów z osobna
        //        // (1.5 * 3 to w sumie ponad 4.5 stopnia płynnego łuku)
        //        if (Mathf.Sign(segAngle) != signNeeded || Mathf.Abs(segAngle) < 1.5f)
        //        {
        //            AddReward(penaltyPerSegment);
        //        }
        //    }
        //}

        //float uprightDot = Vector3.Dot(body.up, Vector3.up);

        //// uprightDot == 1 to idealny pion, 0 to leżenie na boku, -1 to do góry nogami.
        //// Jeśli uprightDot spadnie poniżej zera (agent leży na plecach) lub jest bliski zeru (leży na boku).
        //if (uprightDot < 0.1f)
        //{
        //    // Agent się przewrócił. To jest stan, z którego prawdopodobnie nie wstanie.
        //    AddReward(-1.0f); // Bolesna kara za wywrotkę
        //    EndEpisode();     // Przerywamy epizod
        //    return;
        //}

        //bool isBellyTouching = m_JdController.bodyPartsDict[body].groundContact.touchingGround ||
        //               m_JdController.bodyPartsDict[segment0].groundContact.touchingGround ||
        //               m_JdController.bodyPartsDict[segment1].groundContact.touchingGround ||
        //               m_JdController.bodyPartsDict[segment2].groundContact.touchingGround;

        //if (isBellyTouching)
        //{
        //    // Mała kara za samo otarcie (jak dotychczas)
        //    AddReward(bellyTouchPenalty);

        //    // Zliczamy czas "czołgania się" lub leżenia
        //    m_BellyTouchTimer += Time.fixedDeltaTime;

        //    // Jeśli leży/czołga się zbyt długo (np. > 2 sekundy), uznajemy to za porażkę
        //    if (m_BellyTouchTimer > maxBellyTouchDuration)
        //    {
        //        AddReward(-1.0f); // Duża kara
        //        EndEpisode();     // Zakończenie epizodu - to wybije go ze strategii "leżenia w kulce"
        //        return;
        //    }
        //}
        //else
        //{
        //    m_BellyTouchTimer = 0f; // Agent wstał, resetujemy timer
        //}
    }

    /// <summary>
    /// Update OrientationCube and DirectionIndicator
    /// </summary>
    void UpdateOrientationObjects()
    {
        m_OrientationCube.UpdateOrientation(segment0, m_Target);
        if (m_DirectionIndicator)
        {
            m_DirectionIndicator.MatchOrientation(m_OrientationCube.transform);
        }
    }

    /// <summary>
    ///Returns the average velocity of all of the body parts
    ///Using the velocity of the body only has shown to result in more erratic movement from the limbs
    ///Using the average helps prevent this erratic movement
    /// </summary>
    Vector3 GetAvgVelocity()
    {
        //Vector3 velSum = Vector3.zero;
        //Vector3 avgVel = Vector3.zero;

        ////ALL RBS
        //int numOfRb = 0;
        //foreach (var item in m_JdController.bodyPartsList)
        //{
        //    numOfRb++;
        //    velSum += item.rb.linearVelocity;
        //}

        //avgVel = velSum / numOfRb;
        //return avgVel;

        // Chcemy prędkości w konkretnym kierunku (dot product)
        Vector3 vel = m_JdController.bodyPartsDict[body].rb.linearVelocity;
        vel.y = 0;
        return vel;

    }

    /// <summary>
    /// Normalized value of the difference in actual speed vs goal walking speed.
    /// </summary>
    public float GetMatchingVelocityReward(Vector3 velocityGoal, Vector3 actualVelocity)
    {
        //distance between our actual velocity and goal velocity
        var velDeltaMagnitude = Mathf.Clamp(Vector3.Distance(actualVelocity, velocityGoal), 0, TargetWalkingSpeed);

        //return the value on a declining sigmoid shaped curve that decays from 1 to 0
        //This reward will approach 1 if it matches perfectly and approach zero as it deviates
        return Mathf.Pow(1 - Mathf.Pow(velDeltaMagnitude / TargetWalkingSpeed, 2), 2);
    }

    /// <summary>
    /// Agent touched the target
    /// </summary>
    public void TouchedTarget()
    {
        AddReward(1f);
    }
}

