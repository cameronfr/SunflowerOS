package com.termux;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;

public class SandboxRunner extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Bundle b = getIntent().getExtras();
        String libName = b.getString("libName");
        System.out.println("SandboxRunner: Loading library " + libName);
        System.load(libName);
        System.out.println("SandboxRunner: Library has been loaded");
    }
}
