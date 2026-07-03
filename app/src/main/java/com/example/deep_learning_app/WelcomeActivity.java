package com.example.deep_learning_app;

import android.content.Intent;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

public class WelcomeActivity extends AppCompatActivity {

    private static final long TOTAL_MILLIS = 5000L;

    private TextView skipText;
    private CountDownTimer countDownTimer;
    private boolean hasNavigated;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_welcome);

        skipText = findViewById(R.id.textSkip);
        skipText.setOnClickListener(v -> navigateToMain());

        updateSkipText(5);
        startCountdown();
    }

    private void startCountdown() {
        countDownTimer = new CountDownTimer(TOTAL_MILLIS, 1000L) {
            @Override
            public void onTick(long millisUntilFinished) {
                long secondsLeft = (long) Math.ceil(millisUntilFinished / 1000.0d);
                updateSkipText(secondsLeft);
            }

            @Override
            public void onFinish() {
                navigateToMain();
            }
        };
        countDownTimer.start();
    }

    private void updateSkipText(long secondsLeft) {
        skipText.setText(getString(R.string.skip_countdown, secondsLeft));
    }

    private void navigateToMain() {
        if (hasNavigated) {
            return;
        }
        hasNavigated = true;

        if (countDownTimer != null) {
            countDownTimer.cancel();
        }

        startActivity(new Intent(this, MainActivity.class));
        finish();
    }

    @Override
    protected void onDestroy() {
        if (countDownTimer != null) {
            countDownTimer.cancel();
        }
        super.onDestroy();
    }
}

