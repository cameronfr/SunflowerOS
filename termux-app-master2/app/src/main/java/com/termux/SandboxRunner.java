package com.termux;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;

public class SandboxRunner extends AppCompatActivity {

    private native void loadNativeCode(String path, String funcname);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        System.loadLibrary("sandbox_runner");

        Bundle b = getIntent().getExtras();
        String libName = b.getString("libName");
        System.out.println("SandboxRunner: Loading library " + libName);
        loadNativeCode(libName, "main");
        System.out.println("SandboxRunner: Library has been loaded");
    }
}
