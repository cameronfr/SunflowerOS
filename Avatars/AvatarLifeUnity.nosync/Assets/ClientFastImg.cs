using AsyncIO;
using NetMQ;
using NetMQ.Sockets;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;
using UnityEngine.UI;
using System.Linq;

public class ReceiverFastImg {
  private readonly Thread receiveThread;
  private bool running;

  public ReceiverFastImg() {
    receiveThread = new Thread((object callback) => {
      using (var socket = new PullSocket()) {
        socket.Connect("tcp://localhost:5558");
        var timeout = new TimeSpan(0, 0, 0, 2);

        while (running) {
          byte[] rawImage;
          if (socket.TryReceiveFrameBytes(timeout, out rawImage)) {
            ((Action<byte[]>)callback)(rawImage);
          }
        }
      }
    });
  }

  public void Start(Action<byte[]> callback) {
    running = true;
    receiveThread.Start(callback);
  }

  public void Stop() {
    running = false;
    receiveThread.Join();
  }
}

public class Character : MonoBehaviour {

  private Dictionary<HumanBodyBones, Quaternion> startingRotations;

  public void Start() {
    print("Start called on extended gameobject");
  }

  // public void SaveTPoseRotations(string[] joints) {
  //   startingRotations = new Dictionary<string, Quaternion>();
  //   for (int i=0; i<joints.Length; i++) {
  //     startingRotations[joints[i]] = this.transform.Find(joints[i]).localRotation;
  //   }
  // }
  //
  // public void SetJointAngleFromDefault(string joint, Quaternion rotation) {
  //   Transform t = this.transform.Find(joint);
  //   // sets global (not relative to parent) rotation on joint
  //   t.localRotation = startingRotations[joint] * rotation;
  // }
  public void SaveTPoseRotations(List<HumanBodyBones> joints) {
    startingRotations = new Dictionary<HumanBodyBones, Quaternion>();
    Animator animator = this.GetComponent<Animator>();
    for (int i=0; i<joints.Count; i++) {
      startingRotations[joints[i]] = animator.GetBoneTransform(joints[i]).rotation;
    }
  }

  public void SetJointAngleFromDefault(HumanBodyBones joint, Quaternion rotation) {
    Animator animator = this.GetComponent<Animator>();
    Transform t = animator.GetBoneTransform(joint);
    // sets global (not relative to parent) rotation on joint
    // order maybe should be diff?
    t.rotation = rotation * startingRotations[joint];
  }
}

public class ClientFastImg : MonoBehaviour {
  private readonly ConcurrentQueue<Action> runOnMainThread = new ConcurrentQueue<Action>();
  private ReceiverFastImg receiver;
  private Texture2D tex;
  public RawImage image;

  private Dictionary<int, Character> allCharacters;
  public GameObject[] avatars;
  public List<HumanBodyBones> bonesList;

  public static T[] ArraySlice<T>(T[] orig, int start, int end) {
    if (end < 0) {
      end = orig.Length + end + 1;
    }
    int length = end - start;
    ArraySegment<T> slice = new ArraySegment<T>(orig, start, length);
    return slice.ToArray();
  }

  public void Start() {
    allCharacters = new Dictionary<int, Character>();

    tex = new Texture2D(1280, 720, TextureFormat.RGB24, mipChain: false);
    // tex = new Texture2D(1280/2, 720/2, TextureFormat.RGB24, mipChain: false);
    // tex = new Texture2D(960, 540, TextureFormat.RGB24, mipChain: false);
    image.texture = tex;

    // string[] joints = new string[] {"Armature/Hips/Right leg", "Armature/Hips/Left leg"};
    bonesList = new List<HumanBodyBones>() {
      HumanBodyBones.Hips,
      HumanBodyBones.LeftUpperLeg,
      HumanBodyBones.LeftLowerLeg,
      HumanBodyBones.RightUpperLeg,
      HumanBodyBones.RightLowerLeg,
      HumanBodyBones.Spine,
      HumanBodyBones.Chest,
      HumanBodyBones.Neck,
      HumanBodyBones.Head,
      HumanBodyBones.LeftUpperArm,
      HumanBodyBones.LeftLowerArm,
      HumanBodyBones.RightUpperArm,
      HumanBodyBones.RightLowerArm,
    };

    ForceDotNet.Force();
    receiver = new ReceiverFastImg();
    receiver.Start((byte[] message) => runOnMainThread.Enqueue(() => {
      byte[] data = ArraySlice(message, 1, -1);
      int cmd = (int)message[0];
      print("Recieved command " + cmd);

      if (cmd == 0) {
        // Cmd: background image
        // |texture-byte[1280*720]|
        ScreenCapture.CaptureScreenshot("Screenshots/frame" + System.DateTimeOffset.Now.ToUnixTimeMilliseconds() + ".png");
        tex.LoadRawTextureData(data);
        tex.Apply(updateMipmaps: false);
      } else if (cmd == 1) {
        // Cmd: spawn new avatar with id. have it save its joint positions.
        // |id-int32|
        GameObject randomAvatarTemplate = avatars[UnityEngine.Random.Range(0, avatars.Length)];
        GameObject characterRaw = Instantiate(randomAvatarTemplate, new Vector3(1 * 2.0f, 0, 0), Quaternion.identity);
        Character character = characterRaw.AddComponent<Character>() as Character;

        // Temp test
        // Transform hips = character.transform.Find("Armature/Hips");
        // hips.Rotate(0, 180, 0);

        // character.SaveTPoseRotations(joints);
        character.SaveTPoseRotations(bonesList);
        int key = BitConverter.ToInt32(data, 0);
        allCharacters[key] = character;
      } else if (cmd == 2) {
        // Cmd: move avatar
        // |id-int32|position-float32[3]
        int key = BitConverter.ToInt32(ArraySlice(data, 0, 4), 0);

        var p = new float[3];
        Buffer.BlockCopy(ArraySlice(data, 4, 4+3*4), 0, p, 0, 12);

        Character character = allCharacters[key];
        // Transform hips = character.transform.Find("Armature/Hips");
        Animator animator = character.GetComponent<Animator>();
        Transform hips = animator.GetBoneTransform(HumanBodyBones.Hips);
        hips.position = new Vector3(p[0], p[1], p[2]);

      } else if (cmd == 3) {
        // Cmd: despawn avatar with id
        int key = BitConverter.ToInt32(ArraySlice(data, 0, 4), 0);
        print("Deleting character " + key);
        Character character = allCharacters[key];
        UnityEngine.Object.Destroy(character.gameObject);
        allCharacters.Remove(key);
      } else if (cmd == 4) {
        // Cmd: set avatar scale by setting a specific femur length
        // |id-int32|scale-float32|
        int key = BitConverter.ToInt32(ArraySlice(data, 0, 4), 0);
        float targetScale = BitConverter.ToSingle(ArraySlice(data, 4, 8), 0);
        Character character = allCharacters[key];

        // Transform hips = character.transform.Find("Armature/Hips");
        // Transform rightLeg = character.transform.Find("Armature/Hips/Right leg");
        // Transform rightKnee = character.transform.Find("Armature/Hips/Right leg/Right knee");
        Animator animator = character.GetComponent<Animator>();
        Transform hips = animator.GetBoneTransform(HumanBodyBones.Hips);
        Transform spine = animator.GetBoneTransform(HumanBodyBones.Spine);
        float spineLength = Vector3.Distance(spine.position, hips.position);
        hips.localScale *= (targetScale / spineLength);
        print("Setting scale to " + targetScale);
      } else if (cmd == 5) {
        // Cmd: set avatar poses w/ quaternions
        // |id-int32|rotations-[4*x]|
        int key = BitConverter.ToInt32(ArraySlice(data, 0, 4), 0);

        // int numJoints = 2;
        // var jointAngles = new float[numJoints*4];
        // Buffer.BlockCopy(ArraySlice(data, 4, 4+numJoints*4*4), 0, jointAngles, 0, numJoints*4*4);
        // Character character = allCharacters[key];
        // for (int i=0; i<joints.Length; i++) {
        //   // Transform t = character.transform.Find(joints[i]);
        //   float[] qFloats = ArraySlice(jointAngles, i*4, i*4+4);
        //   Quaternion q = new Quaternion(qFloats[0], qFloats[1], qFloats[2], qFloats[3]);
        //   character.SetJointAngleFromDefault(joints[i], q);
        //   // t.rotation = q;
        // }
        int numJoints = bonesList.Count;
        var jointAngles = new float[numJoints*4];
        Buffer.BlockCopy(ArraySlice(data, 4, 4+numJoints*4*4), 0, jointAngles, 0, numJoints*4*4);
        Character character = allCharacters[key];
        for(int i=0; i<bonesList.Count; i++) {
            float[] qFloats = ArraySlice(jointAngles, i*4, i*4+4);
            Quaternion q = new Quaternion(qFloats[0], qFloats[1], qFloats[2], qFloats[3]);
            print(qFloats[0] + " " + qFloats[1] + " " + qFloats[2] + " " + qFloats[3]);
            character.SetJointAngleFromDefault(bonesList[i], q);
        }
      }
    }));
  }

  public void Update() {
    if (!runOnMainThread.IsEmpty) {
      Action action;
      while (runOnMainThread.TryDequeue(out action)) {
        action.Invoke();
      }
    }
  }

  private void OnDestroy() {
    receiver.Stop();
    // NetMQConfig.Cleanup(false); //hangs on mac for some reason, so don't call
  }
}
