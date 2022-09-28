package com.termux;

import androidx.appcompat.app.AppCompatActivity;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Bundle;

public class SandboxRunner extends AppCompatActivity {

  private native void loadNativeCode(String path, String funcname);

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    System.out.println("SanboxRunner: onCreate");

    // Shutdown when broadcast received.
    // BroadcastReceiver rec = new BroadcastReceiver()
    // {
    //     @Override
    //     public void onReceive(Context context, Intent intent)
    //     {
    //       System.out.println("SandboxRunner: got shutdown msg");
    //       //finish();
    //     }
    // };
    // IntentFilter filter = new IntentFilter("com.termux.app.RUNNER_SHUTDOWN");
    // registerReceiver(rec, filter);

    Bundle b = getIntent().getExtras();
    String libName = b.getString("libName");

    // Thread t = new Thread(new Runnable() {
    //   @Override
    //   public void run() {
        // System.out.println("SandboxRunner: thread starting, libName=" + libName);
    //     // If load the code in new thread, if the code segfaults the activity exits more gracefully.
        // System.loadLibrary("sandbox_runner");
        // loadNativeCode(libName, "main");
        // System.out.println("SandboxRunner: thread done");
    //     finish();
    //   }
    // });
    // System.out.println("SandboxRunner: IN UI thread, starting runner thread " + libName);
    // t.start();
  }

  @Override
  protected void onStart() {
    super.onStart();
    Bundle b = getIntent().getExtras();
    String libName = b.getString("libName");
    System.out.println("SandboxRunner: onStart");
    System.loadLibrary("sandbox_runner");
    loadNativeCode(libName, "main");
    System.out.println("SandboxRunner: thread done");
  }

  @Override
  protected void onDestroy() {
    super.onDestroy();
    // Kill process
    System.out.println("SandboxRunner: onDestroy, killing process");
    android.os.Process.killProcess(android.os.Process.myPid());
  }

  @Override
  protected void onStop() {
    System.out.println("SandboxRunner: onStop, killing process");
    android.os.Process.killProcess(android.os.Process.myPid());
  }

  @Override
  protected void onNewIntent(Intent intent) {
    super.onNewIntent(intent);
    System.out.println("SandboxRunner: onNewIntent");
  }
}
