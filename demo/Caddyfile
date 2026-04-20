# Caddy config for the public NeuroLinked demo server
#
# Handles: automatic HTTPS (Let's Encrypt), gzip, rate limiting
# (via rate_limit module), and reverse-proxy to the brain container.
#
# Usage: set NLKD_DEMO_HOST env var to your domain, then:
#   docker-compose up -d
#
# Replace `demo.neurolinked.ai` below with your actual domain before deploy.

{
    # Global options
    email admin@neurolinked.ai
    # Uncomment for local testing:
    # local_certs
}

demo.neurolinked.ai {
    encode zstd gzip

    # Rate limit — 30 requests per minute per IP at the edge
    @rate_limited {
        method POST GET PUT DELETE
    }
    # Rate limit directive requires the caddy-ratelimit plugin.
    # If you're running stock Caddy without it, comment out the next block
    # and rely on the brain's in-app RateLimiter instead.
    # rate_limit @rate_limited {
    #     zone dynamic
    #     key {client_ip}
    #     events 30
    #     window 1m
    # }

    # Proxy everything to the brain
    reverse_proxy neurolinked:8000 {
        # Pass client IP for in-app rate limiting
        header_up X-Real-IP {remote_host}
        header_up X-Forwarded-For {remote_host}
        header_up X-Forwarded-Proto https
    }

    # Security headers
    header {
        Strict-Transport-Security "max-age=31536000; includeSubDomains"
        X-Content-Type-Options "nosniff"
        X-Frame-Options "DENY"
        Referrer-Policy "same-origin"
    }

    log {
        output file /data/access.log
        format json
    }
}
